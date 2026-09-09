import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_formulation_matches_repository_dense_reference():
    analysis = load_module('pool_analysis', ROOT / 'scripts' / 'run_k_reciprocal_pool_analysis.py')
    reference = load_module('rerank_reference', ROOT / 'torchreid' / 'utils' / 'rerank.py')
    rng = np.random.RandomState(7)
    query = analysis.l2_normalize(rng.randn(4, 8).astype(np.float32))
    gallery = analysis.l2_normalize(rng.randn(9, 8).astype(np.float32))
    combined = np.concatenate((query, gallery))
    k1, k2, lambda_value = 3, 2, .3
    distance = lambda left, right: 1. - np.clip(left.dot(right.T), -1., 1.)
    expected = reference.re_ranking(
        distance(query, gallery), distance(query, query), distance(gallery, gallery),
        k1=k1, k2=k2, lambda_value=lambda_value)

    initial_rank, row_max = analysis.exact_initial_rank(combined, k1 + 1, 16)
    v_matrix = analysis.build_sparse_v(combined, initial_rank, row_max, k1)
    v_matrix = analysis.local_query_expansion(v_matrix, initial_rank, k2)
    v_csc = v_matrix.tocsc()
    actual = np.empty_like(expected)
    query_count = len(query)
    for qi in range(query_count):
        overlap = np.zeros(len(combined), dtype=np.float32)
        start, stop = v_matrix.indptr[qi], v_matrix.indptr[qi + 1]
        for column, q_value in zip(v_matrix.indices[start:stop], v_matrix.data[start:stop]):
            c_start, c_stop = v_csc.indptr[column], v_csc.indptr[column + 1]
            rows = v_csc.indices[c_start:c_stop]
            overlap[rows] += np.minimum(q_value, v_csc.data[c_start:c_stop])
        jaccard = 1. - overlap[query_count:] / (2. - overlap[query_count:])
        raw = distance(query[qi:qi + 1], gallery)[0]
        original = raw * raw / row_max[qi]
        actual[qi] = (1. - lambda_value) * jaccard + lambda_value * original
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    test_sparse_formulation_matches_repository_dense_reference()
    print('sparse k-reciprocal formulation matches dense reference')
