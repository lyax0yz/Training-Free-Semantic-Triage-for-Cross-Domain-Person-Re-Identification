import argparse
import json
import math
import os
import os.path as osp
import textwrap

from PIL import Image, ImageDraw, ImageFont


ATTRIBUTES = ('gender', 'build', 'age', 'upper_colour', 'upper_type',
              'lower_colour', 'lower_type', 'backpack', 'handbag', 'hat')


def font(size, bold=False):
    candidates = ([r'C:\Windows\Fonts\arialbd.ttf', r'C:\Windows\Fonts\segoeuib.ttf']
                  if bold else [r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\segoeui.ttf'])
    for path in candidates:
        if osp.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE_FONT, LABEL_FONT, TEXT_FONT = font(24, True), font(18, True), font(15)


def open_person(path, size=(220, 330)):
    image = Image.open(path).convert('RGB')
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, 'white')
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def attribute_lines(attributes):
    lines = []
    for name in ATTRIBUTES:
        value = attributes.get(name, {})
        if value:
            lines.append('{}={} c={:.2f} a={:.2f}'.format(
                name, value.get('value', 'unknown'), float(value.get('confidence', 0.)),
                float(value.get('vote_agreement', 0.))))
    return lines


def decision_lines(candidate):
    accepted, rejected = [], []
    for item in candidate.get('attribute_decisions', []):
        label = '{}:{} r={:.2f}'.format(
            item['attribute'], item['reason'], float(item.get('reliability', 0.)))
        (accepted if item.get('accepted') else rejected).append(label)
    return ['accepted: ' + (', '.join(accepted) if accepted else 'none'),
            'rejected: ' + (', '.join(rejected) if rejected else 'none')]


def wrapped(lines, width=48):
    output = []
    for line in lines:
        output.extend(textwrap.wrap(line, width=width) or [''])
    return output


def draw_column(canvas, x, label, image_path, attributes, candidate=None):
    draw = ImageDraw.Draw(canvas)
    draw.text((x, 52), label, fill='black', font=LABEL_FONT)
    canvas.paste(open_person(image_path), (x, 82))
    y = 420
    lines = attribute_lines(attributes)
    if candidate is not None:
        lines = [
            'visual={:.5f} sem={:.4f}'.format(
                candidate['visual_similarity'], candidate['semantic_adjustment']),
            'effective={:.4f} quality={:.3f}'.format(
                candidate['effective_semantic_adjustment'], candidate['quality_scale']),
            'final={:.5f}'.format(candidate['final_score'])] + lines + decision_lines(candidate)
    for line in wrapped(lines):
        draw.text((x, y), line, fill='black', font=TEXT_FONT)
        y += 19


def render_case(case, output_path):
    canvas = Image.new('RGB', (1120, 800), '#f2f2f2')
    draw = ImageDraw.Draw(canvas)
    title = ('{} | query {} | true-false visual gap={:+.5f}'.format(
        case['case_type'], case['query_index'], case['true_minus_false_visual_gap']))
    draw.text((24, 16), title, fill='black', font=TITLE_FONT)
    draw_column(canvas, 30, 'QUERY', case['query']['image_path'],
                case['query']['attributes'])
    draw_column(canvas, 390, 'TRUE CANDIDATE', case['true_candidate']['image_path'],
                case['true_candidate']['attributes'], case['true_candidate'])
    draw_column(canvas, 750, 'COMPETING FALSE', case['competing_false_candidate']['image_path'],
                case['competing_false_candidate']['attributes'], case['competing_false_candidate'])
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    return canvas


def contact_sheet(images, output_path, columns=2):
    thumb_size = (560, 400)
    rows = int(math.ceil(len(images) / float(columns)))
    sheet = Image.new('RGB', (columns * thumb_size[0], rows * thumb_size[1]), 'white')
    for index, image in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        sheet.paste(thumb, ((index % columns) * thumb_size[0],
                            (index // columns) * thumb_size[1]))
    sheet.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostic-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cases-per-sheet', type=int, default=8)
    parser.add_argument('--max-per-category', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {'categories': {}}
    for category in ('corrected', 'harmful', 'reachable_not_corrected'):
        path = osp.join(args.diagnostic_dir, category + '.json')
        with open(path, encoding='utf-8') as handle:
            cases = json.load(handle)['cases']
        if args.max_per_category:
            cases = cases[:args.max_per_category]
        category_dir = osp.join(args.output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        rendered = []
        individual_paths = []
        for index, case in enumerate(cases):
            output = osp.join(category_dir, '{:04d}_query_{:04d}.png'.format(
                index + 1, case['query_index']))
            rendered.append(render_case(case, output))
            individual_paths.append(output)
        sheet_paths = []
        for start in range(0, len(rendered), args.cases_per_sheet):
            output = osp.join(args.output_dir, '{}_contact_{:03d}.png'.format(
                category, start // args.cases_per_sheet + 1))
            contact_sheet(rendered[start:start + args.cases_per_sheet], output)
            sheet_paths.append(output)
        manifest['categories'][category] = {
            'count': len(cases), 'individual_pngs': individual_paths,
            'contact_sheets': sheet_paths}
        print('Rendered {} {} cases'.format(len(cases), category), flush=True)
    with open(osp.join(args.output_dir, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == '__main__':
    main()
