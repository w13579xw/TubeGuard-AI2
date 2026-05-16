import os
import csv
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
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

    特点：
    - 自动处理灰度图（转为3通道）
    - 支持训练时数据增强
    - 支持类别权重计算
    """

    LABEL_MAP = {
        '[无缺陷]': 0,
        '[有缺陷]': 1,
        '无缺陷': 0,
        '有缺陷': 1,
    }

    def __init__(self, csv_path, images_dir, split='train', transform=None, image_size=512, augment=False):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')

        if transform is not None:
            self.transform = transform
        elif self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

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
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image_path'])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)

        label = torch.tensor(sample['label'], dtype=torch.long)
        mask = torch.zeros(self.image_size, self.image_size)

        return {
            'image': image,
            'label': label,
            'mask': mask,
            'defect_type': 'defect' if label == 1 else 'good',
            'image_path': sample['image_path'],
        }

    def get_class_counts(self):
        """统计各类别样本数"""
        n_normal = sum(1 for s in self.samples if s['label'] == 0)
        n_defect = sum(1 for s in self.samples if s['label'] == 1)
        return n_normal, n_defect

    def get_sampler(self):
        """
        返回WeightedRandomSampler，用于处理类别不平衡。
        对少数类过采样，使每个epoch中两类样本数量相等。
        """
        labels = [s['label'] for s in self.samples]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[l] for l in labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


def build_dataloader(config, split='train', use_sampler=True):
    """根据配置构建DataLoader。"""
    from torch.utils.data import DataLoader

    data_config = config.get('data', {})
    train_config = config.get('train', {})

    dataset = CSVDataset(
        csv_path=data_config.get(f'{split}_csv', f'data/{split}.csv'),
        images_dir=data_config.get('images_dir', 'data/images'),
        split=split,
        image_size=data_config.get('image_size', 512),
        augment=train_config.get('augment', True),
    )

    batch_size = train_config.get('batch_size', 4) if split == 'train' else 1

    sampler = None
    shuffle = (split == 'train')
    if split == 'train' and use_sampler:
        sampler = dataset.get_sampler()
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=(split == 'train'),
    )

    return loader, dataset


if __name__ == "__main__":
    dataset = CSVDataset(
        csv_path='data/train.csv',
        images_dir='data/images',
        split='train',
        image_size=512,
        augment=True,
    )
    print(f"Dataset size: {len(dataset)}")
    n_normal, n_defect = dataset.get_class_counts()
    print(f"Normal: {n_normal}, Defect: {n_defect}")

    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label: {sample['label']}")