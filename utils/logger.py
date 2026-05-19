import os
import csv
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TrainingLogger:
    """
    训练日志记录器。
    - 将每个epoch的loss和指标写入CSV
    - 同时写入文本日志文件
    - 训练结束后绘制loss曲线和指标曲线
    """

    def __init__(self, log_dir, stage=1):
        self.log_dir = log_dir
        self.stage = stage
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, f'stage{stage}_metrics.csv')
        self.log_path = os.path.join(log_dir, f'stage{stage}_train.log')

        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.log_file = open(self.log_path, 'w', encoding='utf-8')

        self.csv_writer = None
        self.headers = None
        self.history = []

    def _init_csv(self, headers):
        if self.csv_writer is None:
            self.headers = headers
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=headers, extrasaction='ignore')
            self.csv_writer.writeheader()
        else:
            new_cols = [h for h in headers if h not in self.headers]
            if new_cols:
                self.headers.extend(new_cols)
                self.csv_file.close()
                # Rewrite CSV with updated headers
                rows = self.history
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.headers, extrasaction='ignore')
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
                self.csv_file = open(self.csv_path, 'a', newline='', encoding='utf-8')
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.headers, extrasaction='ignore')

    def log_epoch(self, epoch, train_metrics, eval_metrics=None, lr=None, elapsed=None):
        """记录一个epoch的信息。"""
        row = {'epoch': epoch}
        for k, v in train_metrics.items():
            row[f'train_{k}'] = f'{v:.6f}'
        if eval_metrics:
            for k, v in eval_metrics.items():
                row[f'eval_{k}'] = f'{v:.6f}'
        if lr is not None:
            row['lr'] = f'{lr:.2e}'
        if elapsed is not None:
            row['elapsed'] = f'{elapsed:.1f}'

        self._init_csv(list(row.keys()))
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.history.append(row)

        msg = f"[Epoch {epoch}] "
        for k, v in train_metrics.items():
            msg += f"{k}={v:.4f} "
        if eval_metrics:
            msg += "| "
            for k, v in eval_metrics.items():
                msg += f"{k}={v:.4f} "
        if lr is not None:
            msg += f"| lr={lr:.2e} "
        if elapsed is not None:
            msg += f"| {elapsed:.1f}s"

        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)

    def log_message(self, msg):
        """记录普通消息。"""
        self.log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_file.flush()

    def close(self):
        self.csv_file.close()
        self.log_file.close()

    def plot_training_curves(self):
        """读取CSV并绘制训练曲线。"""
        if not self.history:
            return

        epochs = [int(r['epoch']) for r in self.history]

        train_loss_keys = [k for k in self.history[0].keys() if k.startswith('train_loss')]
        eval_keys = [k for k in self.history[0].keys() if k.startswith('eval_')]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Stage {self.stage} Training Curves', fontsize=14)

        ax = axes[0, 0]
        for key in train_loss_keys:
            values = [float(r.get(key, 0)) for r in self.history]
            label = key.replace('train_', '')
            ax.plot(epochs, values, label=label, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if 'train_loss_total' in self.history[0]:
            ax = axes[0, 1]
            values = [float(r.get('train_loss_total', 0)) for r in self.history]
            ax.plot(epochs, values, color='red', linewidth=2, label='train_loss_total')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Total Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

        if eval_keys:
            ax = axes[1, 0]
            auroc_keys = [k for k in eval_keys if 'AUROC' in k or 'F1' in k]
            for key in auroc_keys:
                values = [float(r.get(key, 0)) for r in self.history]
                label = key.replace('eval_', '')
                ax.plot(epochs, values, label=label, linewidth=1.5, marker='o', markersize=3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title('Evaluation Metrics')
            ax.legend()
            ax.grid(True, alpha=0.3)

        if 'lr' in self.history[0]:
            ax = axes[1, 1]
            values = [float(r.get('lr', 0)) for r in self.history]
            ax.plot(epochs, values, color='green', linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, f'stage{self.stage}_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to: {plot_path}")
        return plot_path


def plot_from_csv(csv_path, output_dir=None):
    """从已有CSV文件绘制训练曲线（独立函数，可用于事后分析）。"""
    import csv as csv_mod

    history = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            history.append(row)

    if not history:
        print("No data in CSV")
        return

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    stage = '1' if 'stage1' in csv_path else '2'
    epochs = [int(r['epoch']) for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Stage {stage} Training Curves', fontsize=14)

    train_loss_keys = [k for k in history[0].keys() if k.startswith('train_loss')]

    ax = axes[0, 0]
    for key in train_loss_keys:
        values = [float(r.get(key, 0)) for r in history]
        label = key.replace('train_', '')
        ax.plot(epochs, values, label=label, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if 'train_loss_total' in history[0]:
        ax = axes[0, 1]
        values = [float(r.get('train_loss_total', 0)) for r in history]
        ax.plot(epochs, values, color='red', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Training Loss')
        ax.grid(True, alpha=0.3)

    eval_keys = [k for k in history[0].keys() if k.startswith('eval_')]
    if eval_keys:
        ax = axes[1, 0]
        auroc_keys = [k for k in eval_keys if 'AUROC' in k or 'F1' in k]
        for key in auroc_keys:
            values = [float(r.get(key, 0)) for r in history]
            label = key.replace('eval_', '')
            ax.plot(epochs, values, label=label, linewidth=1.5, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Evaluation Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if 'lr' in history[0]:
        ax = axes[1, 1]
        values = [float(r.get('lr', 0)) for r in history]
        ax.plot(epochs, values, color='green', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'stage{stage}_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {plot_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        plot_from_csv(sys.argv[1])
    else:
        print("Usage: python logger.py <csv_path>")