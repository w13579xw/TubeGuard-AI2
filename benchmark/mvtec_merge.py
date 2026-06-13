"""
Merge per-category MVTec results into final summary table.
Usage: python benchmark/mvtec_merge.py [--output_dir logs/mvtec]
"""
import os, json, argparse

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]

# Texture vs object grouping
TEXTURE = ['carpet', 'grid', 'leather', 'tile', 'wood']
OBJECT = [c for c in CATEGORIES if c not in TEXTURE]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='logs/mvtec')
    args = parser.parse_args()

    all_results = {}
    for cat in CATEGORIES:
        path = os.path.join(args.output_dir, f'{cat}.json')
        if os.path.exists(path):
            with open(path) as f:
                all_results[cat] = json.load(f)

    if not all_results:
        print(f"No results in {args.output_dir}/")
        return

    # Save merged
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    # Print table
    print(f"\n{'='*78}\n  MVTec AD Results ({len(all_results)}/{len(CATEGORIES)} categories)\n{'='*78}")
    print(f"{'Category':<14} {'I-AUROC':>10} {'I-F1max':>10} {'AU-PR':>10} {'P-AUROC':>10} {'PRO':>10}")
    print(f"{'-'*78}")

    def avg(keys, ds):
        out = {}
        for k in keys:
            v = [d.get(k, 0) for d in ds if k in d]
            out[k] = sum(v) / len(v) if v else 0
        return out

    keys = ['I-AUROC', 'I-F1max', 'I-AU-PR', 'P-AUROC', 'PRO']

    # Texture group
    tex_data = [all_results[c] for c in TEXTURE if c in all_results]
    obj_data = [all_results[c] for c in OBJECT if c in all_results]

    if tex_data:
        print("--- Textures ---")
        for cat in TEXTURE:
            if cat not in all_results:
                continue
            r = all_results[cat]
            print(f"{cat:<14} {r['I-AUROC']:>10.4f} {r.get('I-F1max',0):>10.4f} "
                  f"{r.get('I-AU-PR',0):>10.4f} {r.get('P-AUROC',0):>10.4f} {r.get('PRO',0):>10.4f}")
        tex_avg = avg(keys, tex_data)
        print(f"{'  texture-avg':<14} {tex_avg['I-AUROC']:>10.4f} {tex_avg['I-F1max']:>10.4f} "
              f"{tex_avg['I-AU-PR']:>10.4f} {tex_avg['P-AUROC']:>10.4f} {tex_avg['PRO']:>10.4f}")

    if obj_data:
        print("--- Objects ---")
        for cat in OBJECT:
            if cat not in all_results:
                continue
            r = all_results[cat]
            print(f"{cat:<14} {r['I-AUROC']:>10.4f} {r.get('I-F1max',0):>10.4f} "
                  f"{r.get('I-AU-PR',0):>10.4f} {r.get('P-AUROC',0):>10.4f} {r.get('PRO',0):>10.4f}")
        obj_avg = avg(keys, obj_data)
        print(f"{'  object-avg':<14} {obj_avg['I-AUROC']:>10.4f} {obj_avg['I-F1max']:>10.4f} "
              f"{obj_avg['I-AU-PR']:>10.4f} {obj_avg['P-AUROC']:>10.4f} {obj_avg['PRO']:>10.4f}")

    print(f"{'-'*78}")
    all_avg = avg(keys, list(all_results.values()))
    print(f"{'MEAN (ALL)':<14} {all_avg['I-AUROC']:>10.4f} {all_avg['I-F1max']:>10.4f} "
          f"{all_avg['I-AU-PR']:>10.4f} {all_avg['P-AUROC']:>10.4f} {all_avg['PRO']:>10.4f}")
    print(f"{'='*78}\n")


if __name__ == '__main__':
    main()
