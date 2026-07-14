#!/usr/bin/env python3
"""Aggregate grasp_benchmark episode JSONs into a per-method table.

Usage: summarize.py <results_dir> [--csv out.csv]
"""
import argparse
import json
import pathlib
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir')
    ap.add_argument('--csv', default='')
    args = ap.parse_args()

    rows = []
    for p in sorted(pathlib.Path(args.results_dir).glob('*.json')):
        try:
            rows.append(json.loads(p.read_text()))
        except ValueError:
            print(f'skipping unparseable {p}')
    if not rows:
        print('no results found')
        return

    by_method = defaultdict(list)
    for r in rows:
        by_method[r.get('method', '?')].append(r)

    hdr = (f"{'method':<20} {'episodes':>8} {'grasp+lift':>10} {'executed':>8} "
           f"{'plan_s':>7} {'exec_s':>7} {'lift_dz_m':>9}")
    print(hdr)
    print('-' * len(hdr))
    for method, rs in sorted(by_method.items()):
        n = len(rs)
        succ = sum(bool(r.get('success')) for r in rs)
        execd = sum(bool(r.get('executed')) for r in rs)
        plan = [r['plan_time_s'] for r in rs if 'plan_time_s' in r]
        ex = [r['exec_time_s'] for r in rs if 'exec_time_s' in r]
        dz = [r['lift_dz_m'] for r in rs if 'lift_dz_m' in r]
        avg = lambda v: (sum(v) / len(v)) if v else float('nan')  # noqa: E731
        print(f'{method:<20} {n:>8} {succ:>7}/{n:<2} {execd:>8} '
              f'{avg(plan):>7.1f} {avg(ex):>7.1f} {avg(dz):>9.3f}')

    errors = [(r.get('method'), r.get('error')) for r in rows if r.get('error')]
    if errors:
        print('\nerrors:')
        for m, e in errors:
            print(f'  [{m}] {e}')

    if args.csv:
        import csv
        keys = ['method', 'gt_name', 'success', 'executed', 'chosen_index',
                'chosen_label', 'plan_time_s', 'exec_time_s', 'lift_dz_m',
                'task_success', 'error']
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {args.csv}')


if __name__ == '__main__':
    main()
