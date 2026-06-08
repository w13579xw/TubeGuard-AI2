"""
Merge individual ablation JSON files into summary and print table.
Usage: python benchmark/ablation_merge.py
"""
import os, json

output_dir = 'logs/ablation'
all_results = {}

for name in ['full', 'no_slic', 'no_topo', 'no_glpe']:
    path = os.path.join(output_dir, f'{name}.json')
    if os.path.exists(path):
        with open(path) as f:
            all_results[name] = json.load(f)

if not all_results:
    print("No results found. Wait for variants to finish.")
    print(f"  looked in: {output_dir}/")
    print(f"  existing: {os.listdir(output_dir) if os.path.exists(output_dir) else 'dir not found'}")
    exit(0)

# Write merged summary
with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=float)

# Print table
baseline = all_results.get('full', {}).get('I-AUROC', 0)
print(f"\n{'='*75}\n  ABLATION SUMMARY\n{'='*75}")
print(f"{'Variant':<12} {'AUROC':>10} {'F1max':>10} {'AU-PR':>10} {'Δ vs Full':>10} {'Epoch':>8}")
print(f"{'-'*75}")
for name in ['full', 'no_slic', 'no_topo', 'no_glpe']:
    if name not in all_results:
        print(f"{name:<12} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>8}")
        continue
    r = all_results[name]
    delta = r['I-AUROC'] - baseline
    print(f"{name:<12} {r['I-AUROC']:>10.4f} {r['I-F1max']:>10.4f} "
          f"{r.get('I-AU-PR', 0):>10.4f} {delta:>+10.4f} {r['best_epoch']:>8}")
print(f"{'='*75}\nModels: checkpoints/ablation_*.pth\n")