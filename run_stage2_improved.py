"""
一键运行 Stage2 改进版：训练 + 评估 + 对比
"""
import os
import subprocess
import sys

def run_cmd(cmd, desc):
    print(f"\n{'='*80}")
    print(f"▶ {desc}")
    print(f"$ {cmd}")
    print('='*80)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"❌ 命令执行失败: {cmd}")
        sys.exit(1)

def main():
    os.makedirs('logs', exist_ok=True)

    print("🚀 Stage2 改进版 - 开始运行")
    print("="*80)

    # 1. 训练改进版 Stage2
    run_cmd(
        f"python train_stage2_from_stage1.py --config configs/default.yaml "
        f"--stage1_checkpoint checkpoints/stage1_best.pth --device cuda",
        "训练改进版 Stage2（全程冻结 Stage1 + 分数融合）"
    )

    # 2. 评估改进版 Stage2
    run_cmd(
        f"python evaluate_stage2_csv.py --config configs/default.yaml "
        f"--checkpoint checkpoints/stage2_best.pth "
        f"--output_dir logs/stage2_v2_eval --device cuda",
        "评估改进版 Stage2"
    )

    # 3. 打印对比结果
    print("\n" + "="*80)
    print("📊 结果对比")
    print("="*80)

    print("\n✅ Stage1 (基线):")
    if os.path.exists('logs/stage1_test_results.txt'):
        with open('logs/stage1_test_results.txt') as f:
            print(f.read())
    else:
        print("  (暂无 Stage1 结果)")

    print("\n❌ 原版 Stage2:")
    if os.path.exists('logs/stage2_compare/stage2_metrics.json'):
        with open('logs/stage2_compare/stage2_metrics.json') as f:
            print(f.read())
    else:
        print("  (暂无原版 Stage2 结果)")

    print("\n✅ 改进版 Stage2:")
    if os.path.exists('logs/stage2_v2_eval/stage2_metrics.json'):
        with open('logs/stage2_v2_eval/stage2_metrics.json') as f:
            print(f.read())
    else:
        print("  (暂无改进版 Stage2 结果)")

    print("\n" + "="*80)
    print("🎉 Stage2 改进版运行完成！")
    print("="*80)

if __name__ == '__main__':
    main()
