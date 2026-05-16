"""
快速测试脚本：用数据集中的几张图片验证模型前向传播是否有bug。
不做训练，只检查数据加载 → 模型forward → 输出shape是否正确。
"""
import os
import sys
import csv
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import CSVDataset
from models.topovarad import TopoVarAD, TopoVarADConfig


def test_dataset():
    """测试数据集加载"""
    print("=" * 50)
    print("1. Testing Dataset Loading")
    print("=" * 50)

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')

    if not os.path.exists(csv_path):
        print(f"  CSV not found: {csv_path}")
        return None
    if not os.path.isdir(images_dir):
        print(f"  Images dir not found: {images_dir}")
        return None

    dataset = CSVDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        split='train',
        image_size=256,
    )
    print(f"  Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Image path: {sample['image_path']}")

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"\n  Batch image shape: {batch['image'].shape}")
    print(f"  Batch label shape: {batch['label'].shape}")
    print("  Dataset OK!\n")
    return dataset


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 50)
    print("2. Testing Model Forward Pass")
    print("=" * 50)

    config = TopoVarADConfig(
        d_model=64,
        n_tpm_layers=2,
        n_heads=4,
        superpixel_scales=(50, 100),
        rqvae_codebook_size=256,
        rqvae_d_code=16,
        rqvae_n_layers=4,
        tar_n_layers=2,
        tar_n_heads=4,
    )
    model = config.build_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    x = torch.randn(1, 3, 256, 256)

    print("\n  --- Stage 1 ---")
    model.set_stage(1)
    try:
        out1 = model(x)
        print(f"  loss_pixel: {out1['loss_pixel'].item():.4f}")
        print(f"  loss_total: {out1['loss_total'].item():.4f}")
        print(f"  reconstructed shape: {out1['reconstructed'].shape}")
        print(f"  z_global shape: {out1['z_global'].shape}")
        print("  Stage 1 OK!")
    except Exception as e:
        print(f"  Stage 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n  --- Stage 2 ---")
    model.set_stage(2)
    try:
        out2 = model(x)
        print(f"  loss_pixel: {out2['loss_pixel'].item():.4f}")
        print(f"  loss_rqvae: {out2['loss_rqvae'].item():.4f}")
        print(f"  loss_ar: {out2['loss_ar'].item():.4f}")
        print(f"  loss_total: {out2['loss_total'].item():.4f}")
        print(f"  codes shape: {out2['codes'].shape}")
        print(f"  logits shape: {out2['logits'].shape}")
        print("  Stage 2 OK!")
    except Exception as e:
        print(f"  Stage 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n  --- Predict ---")
    try:
        model.eval()
        with torch.no_grad():
            img_scores, px_scores = model.predict(x)
        print(f"  image_scores shape: {img_scores.shape}")
        print(f"  pixel_scores shape: {px_scores.shape}")
        print("  Predict OK!")
    except Exception as e:
        print(f"  Predict FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_end_to_end(dataset):
    """端到端测试：数据集 → DataLoader → 模型"""
    print("=" * 50)
    print("3. Testing End-to-End (Dataset → Model)")
    print("=" * 50)

    if dataset is None:
        print("  Skipped (no dataset)")
        return

    config = TopoVarADConfig(
        d_model=64,
        n_tpm_layers=2,
        n_heads=4,
        superpixel_scales=(50, 100),
        rqvae_codebook_size=256,
        rqvae_d_code=16,
        rqvae_n_layers=4,
        tar_n_layers=2,
        tar_n_heads=4,
    )
    model = config.build_model()
    model.set_stage(1)

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    images = batch['image']

    print(f"  Input batch shape: {images.shape}")
    try:
        out = model(images)
        print(f"  loss_total: {out['loss_total'].item():.4f}")
        print(f"  reconstructed shape: {out['reconstructed'].shape}")
        print("  End-to-End OK!")
    except Exception as e:
        print(f"  End-to-End FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("TopoVarAD Quick Test")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    dataset = test_dataset()
    ok = test_model_forward()
    if ok:
        test_end_to_end(dataset)

    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)