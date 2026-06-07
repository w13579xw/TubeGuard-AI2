"""
Print ablation summary table from merged JSON.
Usage: python benchmark/ablation_merge.py
"""
import os, json

merge_path = 'logs/ablation/summary.json'
if os.path.exists(merge_path):
    with open(merge_path) as f:
        all_results = json.load(f)
    baseline = all_results.get('full', {}).get('I-AUROC', 0)
    print(f"\n{'='*70}\n  ABLATION SUMMARY\n{'='*70}")
    print(f"{'Variant':<12} {'AUROC':>10} {'F1max':>10} {'AU-PR':>10} {'Δ vs Full':>10} {'Epoch':>8}  {'Time':>10}")
    print(f"{'-'*70}")
    for name in ['full', 'no_slic', 'no_topo', 'no_glpe']:
        if name not in all_results:
            continue
        r = all_results[name]
        delta = r['I-AUROC'] - baseline
        t = r.get('train_time_h', 0)
        print(f"{name:<12} {r['I-AUROC']:>10.4f} {r['I-F1max']:>10.4f} "
              f"{r.get('I-AU-PR', 0):>10.4f} {delta:>+10.4f} {r['best_epoch']:>8}  {t:>8.1f}h")
    print(f"{'='*70}\n")
else:
    print(f"Not found: {merge_path} — wait for variants to finish.")