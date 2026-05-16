import os
import csv
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection数据集。
    目录结构:
        root/
        ├── train/
        │   └── good/          # 正常训练样本
        └── test/
            ├── good/          # 正常测试样本
            ├── scratch/       # 异常样本
            ├── crack/
            └── ...
        └── ground_truth/
            ├── scratch/       # 像素级标注
            ├── crack/
            └── ...
    """

    def __init__(self, root, category='bottle', split='train', transform=None, image_size=512):
        self.root = root
        self.category = category
        self.split = split
        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        category_dir = os.path.join(self.root, self.category)

        if self.split == 'train':
            train_dir = os.path.join(category_dir, 'train', 'good')
            if os.path.isdir(train_dir):
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    for path in sorted(glob.glob(os.path.join(train_dir, ext))):
                        self.samples.append({
                            'image_path': path,
                            'label': 0,
                            'mask_path': None,
                            'defect_type': 'good',
                        })
        else:
            test_dir = os.path.join(category_dir, 'test')
            gt_dir = os.path.join(category_dir, 'ground_truth')

            if os.path.isdir(test_dir):
                for defect_type in sorted(os.listdir(test_dir)):
                    defect_dir = os.path.join(test_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue

                    label = 0 if defect_type == 'good' else 1

                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                        for path in sorted(glob.glob(os.path.join(defect_dir, ext))):
                            mask_path = None
                            if label == 1 and gt_dir:
                                basename = os.path.splitext(os.path.basename(path))[0]
                                mask_candidate = os.path.join(gt_dir, defect_type, basename + '_mask.png')
                                if os.path.exists(mask_candidate):
                                    mask_path = mask_candidate

                            self.samples.append({
                                'image_path': path,
                                'label': label,
                                'mask_path': mask_path,
                                'defect_type': defect_type,
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(sample['label'], dtype=torch.long)

        if sample['mask_path'] is not None:
            mask = Image.open(sample['mask_path']).convert('L')
            mask = transforms.Resize((self.image_size, self.image_size))(mask)
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(self.image_size, self.image_size)

        return {
            'image': image,
            'label': label,
            'mask': mask,
            'defect_type': sample['defect_type'],
            'image_path': sample['image_path'],
        }


class CustomIndustrialDataset(Dataset):
    """
    自定义工业数据集。
    目录结构:
        root/
        ├── train/
        │   ├── normal/        # 正常样本
        │   └── abnormal/      # 异常样本（弱监督仅有图像级标签）
        └── test/
            ├── normal/
            └── abnormal/
    """

    def __init__(self, root, split='train', transform=None, image_size=512):
        self.root = root
        self.split = split
        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            return

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            label = 0 if class_name in ('normal', 'good') else 1

            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                for path in sorted(glob.glob(os.path.join(class_dir, ext))):
                    self.samples.append({
                        'image_path': path,
                        'label': label,
                        'defect_type': class_name,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(sample['label'], dtype=torch.long)
        mask = torch.zeros(self.image_size, self.image_size)

        return {
            'image': image,
            'label': label,
            'mask': mask,
            'defect_type': sample['defect_type'],
            'image_path': sample['image_path'],
        }


class CSVDataset(Dataset):
    """
    CSV格式工业数据集。
    CSV格式: image,label
        image列为文件名（如 2506.jpg）
        label列为 [有缺陷] 或 [无缺陷]
    图像存放在 images_dir 目录下。
    """

    LABEL_MAP = {
        '[无缺陷]': 0,
        '[有缺陷]': 1,
        '无缺陷': 0,
        '有缺陷': 1,
    }

    def __init__(self, csv_path, images_dir, split='train', transform=None, image_size=512):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.split = split
        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.samples = []
        self._load_csv()

    def _load_csv(self):
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['image'].strip()
                label_str = row['label'].strip()
                label = self.LABEL_MAP.get(label_str, 0)

                image_path = os.path.join(self.images_dir, filename)
                if os.path.exists(image_path):
                    self.samples.append({
                        'image_path': image_path,
                        'label': label,
                        'label_str': label_str,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(sample['label'], dtype=torch.long)
        mask = torch.zeros(self.image_size, self.image_size)

        return {
            'image': image,
            'label': label,
            'mask': mask,
            'defect_type': sample['label_str'],
            'image_path': sample['image_path'],
        }


def build_dataloader(config, split='train'):
    """根据配置构建DataLoader。支持MVTec和CSV两种数据格式。"""
    from torch.utils.data import DataLoader

    csv_path = config.get('csv_path')
    if csv_path:
        images_dir = config.get('images_dir', 'data/images')
        dataset = CSVDataset(
            csv_path=csv_path,
            images_dir=images_dir,
            split=split,
            image_size=config.get('image_size', 512),
        )
    else:
        dataset = MVTecDataset(
            root=config.get('dataset_path', 'data/mvtec'),
            category=config.get('category', 'bottle'),
            split=split,
            image_size=config.get('image_size', 512),
        )

    loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 4) if split == 'train' else 1,
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=(split == 'train'),
    )

    return loader


if __name__ == "__main__":
    dataset = MVTecDataset(
        root='data/mvtec',
        category='bottle',
        split='train',
        image_size=512,
    )
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Mask shape: {sample['mask'].shape}")
