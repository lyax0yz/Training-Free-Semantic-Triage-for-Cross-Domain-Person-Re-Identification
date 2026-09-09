from __future__ import print_function

import argparse
import json
import os
import os.path as osp
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import torchreid
from default_config import get_default_config, imagedata_kwargs
from torchreid import metrics
from torchreid.utils import check_isfile, load_pretrained_weights, set_random_seed


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export ReID features, metadata and query-gallery distances'
    )
    parser.add_argument('--config-file', type=str, default='')
    parser.add_argument('-s', '--sources', type=str, nargs='+')
    parser.add_argument('-t', '--targets', type=str, nargs='+')
    parser.add_argument('--transforms', type=str, nargs='+')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument(
        '--output', required=True, help='Output .npz artifact path'
    )
    parser.add_argument(
        '--distance-chunk-size', type=int, default=128,
        help='Number of query rows per distance-matrix chunk'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Allow replacing an existing output file'
    )
    parser.add_argument(
        'opts', default=None, nargs=argparse.REMAINDER,
        help='Configuration overrides, e.g. model.load_weights PATH'
    )
    return parser.parse_args()


def extract_split_features(model, data_loader, use_gpu, split_name):
    features, pids, camids, dsetids, impaths = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            imgs = data['img']
            if use_gpu:
                imgs = imgs.cuda(non_blocking=True)
            batch_features = model(imgs)
            if isinstance(batch_features, (tuple, list)):
                batch_features = batch_features[-1]
            features.append(batch_features.detach().cpu())
            pids.extend(data['pid'].tolist())
            camids.extend(data['camid'].tolist())
            dsetids.extend(data['dsetid'].tolist())
            impaths.extend(data['impath'])
            if (batch_idx + 1) % 100 == 0:
                print('{}: processed {} batches'.format(split_name, batch_idx + 1))

    features = torch.cat(features, dim=0).numpy().astype(np.float32, copy=False)
    return {
        'features': features,
        'pids': np.asarray(pids, dtype=np.int64),
        'camids': np.asarray(camids, dtype=np.int64),
        'dsetids': np.asarray(dsetids, dtype=np.int64),
        'impaths': np.asarray(impaths, dtype=np.str_)
    }


def compute_distance_matrix_chunked(qf, gf, metric, chunk_size, temp_dir):
    num_query, num_gallery = qf.shape[0], gf.shape[0]
    fd, temp_path = tempfile.mkstemp(
        prefix='torchreid_distmat_', suffix='.npy', dir=temp_dir
    )
    os.close(fd)
    distmat = np.lib.format.open_memmap(
        temp_path, mode='w+', dtype=np.float32, shape=(num_query, num_gallery)
    )
    gf_tensor = torch.from_numpy(gf)
    try:
        for start in range(0, num_query, chunk_size):
            end = min(start + chunk_size, num_query)
            qf_tensor = torch.from_numpy(qf[start:end])
            distances = metrics.compute_distance_matrix(qf_tensor, gf_tensor, metric)
            distmat[start:end] = distances.numpy()
            print('Distance rows {}/{}'.format(end, num_query))
        distmat.flush()
        return temp_path
    except Exception:
        del distmat
        if osp.exists(temp_path):
            os.remove(temp_path)
        raise


def main():
    args = parse_args()
    if args.distance_chunk_size < 1:
        raise ValueError('--distance-chunk-size must be positive')

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)

    if cfg.data.type != 'image':
        raise ValueError('This experimental exporter currently supports image ReID only')
    if len(cfg.data.targets) != 1:
        raise ValueError('Export one target dataset at a time; got {}'.format(cfg.data.targets))
    if not cfg.model.load_weights or not check_isfile(cfg.model.load_weights):
        raise FileNotFoundError('A valid model.load_weights checkpoint is required')

    output_path = osp.abspath(osp.expanduser(args.output))
    if not output_path.endswith('.npz'):
        output_path += '.npz'
    if osp.exists(output_path) and not args.overwrite:
        raise FileExistsError('{} already exists; pass --overwrite to replace it'.format(output_path))
    output_dir = osp.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print('Building data manager ...')
    datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    target = cfg.data.targets[0]

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    load_pretrained_weights(model, cfg.model.load_weights)
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    loaders = datamanager.test_loader[target]
    print('Extracting query features ...')
    query = extract_split_features(model, loaders['query'], cfg.use_gpu, 'query')
    print('Extracting gallery features ...')
    gallery = extract_split_features(model, loaders['gallery'], cfg.use_gpu, 'gallery')

    if cfg.test.normalize_feature:
        print('L2-normalizing features before distance computation ...')
        query['features'] = F.normalize(torch.from_numpy(query['features']), p=2, dim=1).numpy()
        gallery['features'] = F.normalize(torch.from_numpy(gallery['features']), p=2, dim=1).numpy()

    estimated_gib = query['features'].shape[0] * gallery['features'].shape[0] * 4 / 1024**3
    print('Computing {:.2f} GiB float32 distance matrix in chunks ...'.format(estimated_gib))
    temp_dist_path = compute_distance_matrix_chunked(
        query['features'], gallery['features'], cfg.test.dist_metric,
        args.distance_chunk_size, output_dir or None
    )
    try:
        distmat = np.load(temp_dist_path, mmap_mode='r')
        metadata = {
            'model_name': cfg.model.name,
            'checkpoint': osp.abspath(cfg.model.load_weights),
            'source_datasets': list(cfg.data.sources),
            'target_dataset': target,
            'distance_metric': cfg.test.dist_metric,
            'normalize_feature': bool(cfg.test.normalize_feature),
            'feature_dim': int(query['features'].shape[1])
        }
        print('Writing {}'.format(output_path))
        np.savez(
            output_path,
            query_features=query['features'],
            gallery_features=gallery['features'],
            query_impaths=query['impaths'],
            gallery_impaths=gallery['impaths'],
            query_pids=query['pids'],
            gallery_pids=gallery['pids'],
            query_camids=query['camids'],
            gallery_camids=gallery['camids'],
            query_dsetids=query['dsetids'],
            gallery_dsetids=gallery['dsetids'],
            distmat=distmat,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True))
        )
    finally:
        if 'distmat' in locals():
            mmap = getattr(distmat, '_mmap', None)
            if mmap is not None:
                mmap.close()
            del distmat
        if osp.exists(temp_dist_path):
            os.remove(temp_dist_path)

    print('Export complete: {}'.format(output_path))


if __name__ == '__main__':
    main()
