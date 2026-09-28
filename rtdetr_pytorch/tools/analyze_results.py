"""Read experiment logs and trusted checkpoints without modifying them."""
import argparse
import csv
import json
from pathlib import Path

METRICS = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']


def read_log(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty log: {path}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline-log', 'sgf-log', 'checkpoint', 'baseline-eval', 'sgf-eval', 'annotations', 'csv']:
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if bool(args.baseline_log) != bool(args.sgf_log):
        parser.error('Provide both --baseline-log and --sgf-log')
    if args.csv and not args.baseline_log:
        parser.error('--csv requires both logs')
    if bool(args.baseline_eval) != bool(args.sgf_eval) or (args.baseline_eval and not args.annotations):
        parser.error('Class AP requires both eval files and --annotations')
    if not any([args.baseline_log, args.checkpoint, args.baseline_eval]):
        parser.error('Provide logs, a checkpoint, or eval files; see --help')
    if args.baseline_log:
        logs = [read_log(p) for p in [args.baseline_log, args.sgf_log]]
        print('Final epoch comparison (AP percent; delta in percentage points)')
        for label, rows in zip(['baseline', 'sgf'], logs):
            print(label, 'completed epochs:', len(rows), 'final epoch:', rows[-1]['epoch'] + 1)
            for index, name in [(0, 'AP'), (3, 'AP_small')]:
                best = max(rows, key=lambda r: r['test_coco_eval_bbox'][index])
                print(f"  best {name}: {best['test_coco_eval_bbox'][index]*100:.3f}, epoch {best['epoch']+1}")
        for i, name in enumerate(METRICS):
            a, b = [rows[-1]['test_coco_eval_bbox'][i] * 100 for rows in logs]
            print(f'{name:12} {a:9.3f} {b:9.3f} delta={b-a:+.3f}')
        if args.csv:
            with args.csv.open('x', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['model', 'epoch', *METRICS, 'train_loss'])
                for label, rows in zip(['baseline', 'sgf'], logs):
                    for row in rows:
                        writer.writerow([label, row['epoch'] + 1,
                                         *[v*100 for v in row['test_coco_eval_bbox'][:6]], row['train_loss']])
    if args.checkpoint or args.baseline_eval:
        import torch
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        for key in ['model', 'ema']:
            if key not in checkpoint:
                continue
            state = checkpoint[key]['module'] if key == 'ema' else checkpoint[key]
            name = 'encoder.sgf.residual_scale'
            print(f'{key} lambda:', state[name].item() if name in state else 'no SGF parameter')
        del checkpoint
    if args.baseline_eval:
        categories = {c['id']: c['name'] for c in json.loads(args.annotations.read_text())['categories']}
        def class_ap(path):
            ev = torch.load(path, map_location='cpu', weights_only=False)
            result = {}
            for i, cat in enumerate(ev['params'].catIds):
                values = []
                for area in ['all', 'small']:
                    a = ev['params'].areaRngLbl.index(area)
                    values_array = ev['precision'][:, :, i, a, -1]
                    valid = values_array[values_array >= 0]
                    values.append(float(valid.mean()*100) if valid.size else float('nan'))
                result[cat] = values
            return result
        baseline, sgf = class_ap(args.baseline_eval), class_ap(args.sgf_eval)
        if baseline.keys() != sgf.keys():
            raise ValueError('Evaluation category sets differ')
        print('category             baseline_AP SGF_AP delta_AP baseline_APs SGF_APs delta_APs')
        for category, (ap, aps) in baseline.items():
            new_ap, new_aps = sgf[category]
            print(f'{categories[category]:20} {ap:8.3f} {new_ap:8.3f} {new_ap-ap:+8.3f}'
                  f' {aps:8.3f} {new_aps:8.3f} {new_aps-aps:+8.3f}')


if __name__ == '__main__':
    main()
