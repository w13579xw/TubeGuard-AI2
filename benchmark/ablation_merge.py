"""
Merge individual ablation result files into a summary JSON and print table.
Run after all ablation variants complete.
Usage: python benchmark/ablation_merge.py --output_dir logs/ablation
"""
import os, sys, json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='logs/ablation')
    args = parser.parse_args()

    all_results = {}
    for name in ['full', 'no_slic', 'no_topo', 'no_glpe']:
        path = os.path.join(args.output_dir, f'{name}.json')
        if os.path.exists(path):
            with open(path) as f:
                all_results[name] = json.load(f)

    if not all_results:
        print("No results found.")
        return

    # Save merged
    merge_path = os.path.join(args.output_dir, 'summary.json')
    with open(merge_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    # Print table
    baseline = all_results.get('full', {}).get('I-AUROC', 0)
    print(f"\n{'='*65}\n  ABLATION SUMMARY\n{'='*65}")
    print(f"{'Variant':<12} {'AUROC':>10} {'F1max':>10} {'AU-PR':>10} {'ΔAUROC':>10} {'Epoch':>8}")
    print(f"{'-'*65}")
    for name in ['full', 'no_slic', 'no_topo', 'no_glpe']:
        if name not in all_results:
            continue
        r = all_results[name]
        delta = r['I-AUROC'] - baseline
        print(f"{name:<12} {r['I-AUROC']:>10.4f} {r['I-F1max']:>10.4f} "
              f"{r.get('I-AU-PR', 0):>10.4f} {delta:>+10.4f} {r['best_epoch']:>8}")
    print(f"{'='*65}\n")
    print(f"Merged: {merge_path}")


if __name__ == '__main__':
    main()