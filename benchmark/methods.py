"""
SOTA anomaly detection baseline implementations for comparison with TopoVarAD.

Supported methods:
  - PaDiM: Patch Distribution Modeling (Defard et al., ICPR 2021)
  - PatchCore: Coreset-based nearest-neighbor (Roth et al., CVPR 2022)
  - AE: Autoencoder baseline (Bergmann et al., VISIGRAPP 2019)
  - EfficientAD: Lightweight teacher-student (simplified) (Batzner et al., WACV 2024)

All methods use the same data loading and evaluation protocol as TopoVarAD.
"""

import os
import numpy as np
import math
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tv_models
from scipy.ndimage import gaussian_filter
from sklearn.random_projection import SparseRandomProjection

from data.dataset import CSVDataset
from utils.metrics import compute_auroc, compute_f1_max, compute_auprc


# ============================================================
# Feature extractors
# ============================================================

def _safe_load_model(model_name, device):
    """
    Load a pre-trained model with download fallback.
    On servers without internet, falls back to locally cached models.
    """
    print(f"  Loading {model_name}...", end=" ")
    if model_name == 'resnet18':
        try:
            model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            print("(download failed, trying local cache...) ", end="")
            model = tv_models.resnet18(weights=None)
    elif model_name == 'wideresnet50':
        try:
            model = tv_models.wide_resnet50_2(
                weights=tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        except Exception:
            print("(download failed, falling back to resnet18...) ", end="")
            model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            model_name = 'resnet18_fallback'
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device).eval()
    print("OK")
    return model, model_name


class ResNet18FeatureExtractor(nn.Module):
    """Extract multi-layer features from ResNet-18."""
    def __init__(self, device='cpu'):
        super().__init__()
        resnet, _ = _safe_load_model('resnet18', device)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return [x1, x2, x3]  # skip layer0 to save memory


class WideResNet50FeatureExtractor(nn.Module):
    """Extract multi-layer features from WideResNet-50-2 (with fallback to ResNet-18)."""
    def __init__(self, device='cpu'):
        super().__init__()
        model, self._actual_model = _safe_load_model('wideresnet50', device)
        self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return [x2, x3]  # PatchCore uses deeper layers


def get_layers_for_backbone(backbone):
    """Get intermediate layer outputs for a given backbone."""
    if backbone == 'resnet18':
        return ['layer1', 'layer2', 'layer3']
    elif backbone == 'wideresnet50':
        return ['layer2', 'layer3']
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# ============================================================
# PaDiM
# ============================================================

class PaDiM:
    """
    PaDiM: Patch Distribution Modeling for Anomaly Detection.
    Fits multivariate Gaussian distributions to patch features and uses
    Mahalanobis distance as anomaly score.

    Memory-efficient: features are pooled to a small spatial grid (16x16)
    and computed online via Welford's algorithm for mean and covariance.

    Reference: Defard et al., ICPR 2021
    """

    def __init__(self, backbone='resnet18', device='cuda', pool_size=16, max_train_samples=800):
        self.device = torch.device(device)
        self.backbone = backbone
        self.pool_size = pool_size
        self.max_train_samples = max_train_samples
        self.model = ResNet18FeatureExtractor(device=self.device).to(self.device).eval()

    @torch.no_grad()
    def fit(self, train_loader):
        """Fit multivariate Gaussian distributions using Welford's online algorithm."""
        print(f"PaDiM: fitting (pool={self.pool_size}, max_samples={self.max_train_samples})...")

        # Online mean/covariance accumulators per layer
        stats = {}  # {layer_idx: {'n': int, 'mean': array, 'M2': array}}
        total_images = 0

        for batch in tqdm(train_loader, desc='PaDiM: extracting'):
            images = batch['image'].to(self.device)
            feats = self.model(images)

            # Pool all layers to same small spatial grid
            for i, f in enumerate(feats):
                f_pooled = F.adaptive_avg_pool2d(f, (self.pool_size, self.pool_size))  # (B, C, Hp, Wp)
                B, C, Hp, Wp = f_pooled.shape
                # Randomly subsample spatial positions to save memory
                n_positions = Hp * Wp
                n_sample_pos = min(n_positions, 64)  # sample 64 positions per image
                rand_idx = torch.randperm(n_positions, device=self.device)[:n_sample_pos]

                f_flat = f_pooled.reshape(B, C, -1)[:, :, rand_idx]  # (B, C, n_sample_pos)
                f_flat = f_flat.permute(0, 2, 1).reshape(-1, C)      # (B*n_sample_pos, C)

                # Welford's online algorithm
                f_np = f_flat.cpu().numpy()
                batch_n = f_np.shape[0]
                batch_mean = f_np.mean(axis=0)
                batch_centered = f_np - batch_mean

                if i not in stats:
                    stats[i] = {'n': batch_n, 'mean': batch_mean,
                                'M2': batch_centered.T @ batch_centered}
                else:
                    # Merge with existing stats (Chan's method)
                    delta = batch_mean - stats[i]['mean']
                    M2_new = batch_centered.T @ batch_centered
                    stats[i]['M2'] = stats[i]['M2'] + M2_new + \
                        delta[:, None] @ delta[None, :] * (stats[i]['n'] * batch_n / (stats[i]['n'] + batch_n))
                    stats[i]['mean'] = (stats[i]['n'] * stats[i]['mean'] + batch_n * batch_mean) / (stats[i]['n'] + batch_n)
                    stats[i]['n'] += batch_n

            total_images += images.shape[0]
            if total_images >= self.max_train_samples:
                break

        # Compute inverse covariances
        self.means = {}
        self.inv_covs = {}
        reg = 0.01  # Regularization

        for layer_idx, s in stats.items():
            cov = s['M2'] / (s['n'] - 1)
            cov_reg = cov + reg * np.eye(cov.shape[0])
            self.means[layer_idx] = s['mean'].reshape(1, -1)
            try:
                self.inv_covs[layer_idx] = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                self.inv_covs[layer_idx] = np.linalg.pinv(cov_reg)
            print(f"  Layer {layer_idx}: dim={cov_reg.shape[0]}, n={s['n']}")

        return self

    @torch.no_grad()
    def predict(self, loader):
        """Compute Mahalanobis distance for each test sample."""
        self.model.eval()
        image_scores = []
        pixel_scores_list = []
        all_labels = []
        Hp = Wp = self.pool_size

        for batch in tqdm(loader, desc='PaDiM: predicting'):
            images = batch['image'].to(self.device)
            labels = batch['label']
            feats = self.model(images)

            batch_mahalanobis = np.zeros((images.shape[0], Hp, Wp))
            for i, f in enumerate(feats):
                if i not in self.means:
                    continue
                f_pooled = F.adaptive_avg_pool2d(f, (Hp, Wp))
                B, C, _, _ = f_pooled.shape
                f_flat = f_pooled.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C).cpu().numpy()
                f_centered = f_flat - self.means[i]
                mahalanobis = np.sum(f_centered * (f_centered @ self.inv_covs[i]), axis=1)
                batch_mahalanobis += mahalanobis.reshape(B, Hp, Wp)

            img_score = batch_mahalanobis.reshape(images.shape[0], -1).max(axis=1)
            image_scores.extend(img_score.tolist())
            all_labels.extend(labels.tolist())

            for j in range(images.shape[0]):
                pmap = torch.tensor(batch_mahalanobis[j]).unsqueeze(0).unsqueeze(0).float()
                pmap = F.interpolate(pmap, size=(512, 512), mode='bilinear', align_corners=False)
                pixel_scores_list.append(pmap.squeeze().cpu().numpy())

        return np.array(image_scores), pixel_scores_list, np.array(all_labels)


# ============================================================
# PatchCore
# ============================================================

class PatchCore:
    """
    PatchCore: Coreset-based nearest-neighbor anomaly detection.
    Constructs a maximally representative coreset of normal patch features
    and uses nearest-neighbor distance as anomaly score.

    Reference: Roth et al., CVPR 2022
    """

    def __init__(self, backbone='wideresnet50', coreset_ratio=0.01, device='cuda',
                 pool_size=16, max_train_samples=500):
        self.device = torch.device(device)
        self.coreset_ratio = coreset_ratio
        self.pool_size = pool_size
        self.max_train_samples = max_train_samples
        self.extractor = WideResNet50FeatureExtractor(device=self.device).to(self.device).eval()

    @torch.no_grad()
    def _extract_features(self, loader):
        """Extract patch features from training data (memory-efficient)."""
        all_features = []
        total_images = 0

        for batch in tqdm(loader, desc='PatchCore: extracting features'):
            images = batch['image'].to(self.device)
            feats = self.extractor(images)

            # Pool to small spatial grid
            pooled_feats = []
            for f in feats:
                f = F.adaptive_avg_pool2d(f, (self.pool_size, self.pool_size))
                pooled_feats.append(f)
            combined = torch.cat(pooled_feats, dim=1)

            B, C, H, W = combined.shape
            # Subsample spatial positions
            n_positions = H * W
            n_sample = min(n_positions, 32)
            rand_idx = torch.randperm(n_positions, device=self.device)[:n_sample]
            combined_flat = combined.reshape(B, C, -1)[:, :, rand_idx]
            patches = combined_flat.permute(0, 2, 1).reshape(-1, C)
            all_features.append(patches.cpu())

            total_images += images.shape[0]
            if total_images >= self.max_train_samples:
                break

        features = torch.cat(all_features, dim=0)
        # Further random subsample if too many
        if features.shape[0] > 100000:
            idx = torch.randperm(features.shape[0])[:100000]
            features = features[idx]
        print(f"  Extracted {features.shape[0]} features (dim={features.shape[1]})")
        return features

    def _greedy_coreset(self, features, n_coreset):
        """Greedy coreset selection as described in PatchCore paper."""
        print(f"  Building coreset: {features.shape[0]} -> {n_coreset} samples")
        n_total = features.shape[0]

        # If few enough features, no need for coreset
        if n_total <= n_coreset:
            return features

        # Randomly sample initial point
        coreset_indices = [np.random.randint(0, n_total)]
        min_distances = torch.norm(features - features[coreset_indices[0]], dim=1)

        pbar = tqdm(range(1, n_coreset), desc='  Greedy coreset', leave=False)
        for _ in pbar:
            # Select farthest point
            new_idx = torch.argmax(min_distances).item()
            coreset_indices.append(new_idx)

            # Update minimum distances
            distances = torch.norm(features - features[new_idx], dim=1)
            min_distances = torch.min(min_distances, distances)

            pbar.set_postfix({'max_dist': f'{min_distances.max().item():.3f}'})

        return features[coreset_indices]

    @torch.no_grad()
    def fit(self, train_loader):
        """Extract features and build coreset."""
        print("PatchCore: extracting training features...")
        features = self._extract_features(train_loader)

        # Normalize features
        self.mean = features.mean(dim=0, keepdim=True)
        self.std = features.std(dim=0, keepdim=True) + 1e-6
        features = (features - self.mean) / self.std

        # Build coreset
        n_coreset = max(100, int(features.shape[0] * self.coreset_ratio))
        self.coreset = self._greedy_coreset(features, n_coreset)
        print(f"  Coreset built: {self.coreset.shape[0]} features")

        return self

    @torch.no_grad()
    def predict(self, loader, k=5):
        """Compute nearest-neighbor distance for anomaly scoring."""
        coreset = self.coreset.to(self.device)
        image_scores = []
        pixel_scores_list = []
        all_labels = []

        for batch in tqdm(loader, desc='PatchCore: predicting'):
            images = batch['image'].to(self.device)
            labels = batch['label']
            feats = self.extractor(images)

            # Combine features
            pooled_feats = []
            target_size = feats[0].shape[2:]
            for f in feats:
                f = F.adaptive_avg_pool2d(f, target_size)
                pooled_feats.append(f)
            combined = torch.cat(pooled_feats, dim=1)

            B, C, H, W = combined.shape
            patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C)
            patches_norm = (patches - self.mean.to(self.device)) / self.std.to(self.device)

            # Compute distances to coreset (in chunks to avoid OOM)
            chunk_size = 4096
            min_dists = torch.full((B * H * W,), float('inf'), device=self.device)

            for i in range(0, len(coreset), chunk_size):
                chunk = coreset[i:i + chunk_size]
                dists = torch.cdist(patches_norm, chunk)  # (B*H*W, chunk_size)
                chunk_min = dists.min(dim=1).values
                min_dists = torch.min(min_dists, chunk_min)

            anomaly_map = min_dists.reshape(B, H, W)

            # Image-level score
            img_score = anomaly_map.reshape(B, -1).max(dim=1).values
            image_scores.extend(img_score.cpu().tolist())
            all_labels.extend(labels.tolist())

            # Pixel-level score
            for j in range(B):
                pmap = anomaly_map[j].unsqueeze(0).unsqueeze(0).float()
                pmap = F.interpolate(pmap, size=(512, 512), mode='bilinear', align_corners=False)
                pmap_smooth = gaussian_filter(pmap.squeeze().cpu().numpy(), sigma=4)
                pixel_scores_list.append(pmap_smooth)

        return np.array(image_scores), pixel_scores_list, np.array(all_labels)


# ============================================================
# Autoencoder baseline
# ============================================================

class AutoencoderBaseline(nn.Module):
    """Simple convolutional autoencoder for anomaly detection."""

    def __init__(self, latent_dim=256):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, latent_dim, 4, 2, 1), nn.BatchNorm2d(latent_dim), nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def anomaly_score(self, x):
        """Reconstruction error as anomaly score."""
        x_hat, _ = self.forward(x)
        # Per-pixel L1 error
        error = torch.abs(x - x_hat).mean(dim=1)  # (B, H, W)
        # Image-level: mean error
        img_score = error.reshape(x.shape[0], -1).mean(dim=1)
        return img_score, error


# ============================================================
# RD4AD: Reverse Distillation for Anomaly Detection
# (Deng & Li, CVPR 2022)
# ============================================================

class RD4ADTeacher(nn.Module):
    """Pre-trained ResNet-18 encoder as teacher."""

    def __init__(self, device='cpu'):
        super().__init__()
        resnet, _ = _safe_load_model('resnet18', device)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        return [f1, f2, f3]


class RD4ADStudent(nn.Module):
    """Decoder that reconstructs teacher features in reverse order."""

    def __init__(self):
        super().__init__()
        # Bottleneck compression
        self.bottleneck = nn.Conv2d(256, 512, 1)

        # Decoder (reverse of ResNet layers)
        self.dec2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
        )

    def forward(self, f3):
        b = self.bottleneck(f3)
        d2 = self.dec2(b)   # reconstruct f2
        d1 = self.dec1(d2)  # reconstruct f1
        return [d1, d2]


class RD4AD:
    """
    RD4AD: Reverse Distillation for Anomaly Detection.
    Teacher (frozen ResNet-18) → Student decoder reconstructs features.
    Anomaly = cosine distance between teacher and student feature maps.

    Reference: Deng & Li, CVPR 2022
    """

    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.teacher = RD4ADTeacher(device=self.device).to(self.device).eval()
        self.student = RD4ADStudent().to(self.device)

    def fit(self, train_loader, epochs=60, lr=0.005):
        """Train student decoder to reconstruct teacher features."""
        self.teacher.eval()
        self.student.train()
        optimizer = optim.AdamW(self.student.parameters(), lr=lr, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print(f"RD4AD: training student ({epochs} epochs, lr={lr})")
        pbar = tqdm(range(epochs), desc='RD4AD training')
        best_loss = float('inf')

        for epoch in pbar:
            total_loss = 0.0
            for batch in train_loader:
                images = batch['image'].to(self.device)
                with torch.no_grad():
                    teacher_feats = self.teacher(images)  # [f1, f2, f3]
                student_feats = self.student(teacher_feats[-1])  # [d1, d2]

                # MSE loss on all feature levels
                loss = F.mse_loss(student_feats[0], teacher_feats[0]) + \
                       F.mse_loss(student_feats[1], teacher_feats[1])
                # Add cosine distance loss for better alignment
                T1 = F.normalize(teacher_feats[0].flatten(1), dim=1)
                S1 = F.normalize(student_feats[0].flatten(1), dim=1)
                loss = loss + 0.5 * (1 - (T1 * S1).sum(dim=1)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'best': f'{best_loss:.4f}'})

        return self

    @torch.no_grad()
    def predict(self, loader):
        """Compute feature discrepancy as anomaly score."""
        self.teacher.eval()
        self.student.eval()
        image_scores = []
        pixel_scores_list = []
        all_labels = []

        for batch in tqdm(loader, desc='RD4AD: predicting'):
            images = batch['image'].to(self.device)
            labels = batch['label']
            teacher_feats = self.teacher(images)
            student_feats = self.student(teacher_feats[-1])

            # Cosine distance at each level
            T = F.normalize(teacher_feats[0].flatten(2), dim=1)  # (B, C, HW)
            S = F.normalize(student_feats[0].flatten(2), dim=1)
            cos_dist = 1 - (T * S).sum(dim=1)  # (B, HW)

            B, C, H, W = teacher_feats[0].shape
            anomaly_map = cos_dist.reshape(B, H, W)

            img_score = anomaly_map.reshape(B, -1).max(dim=1).values
            image_scores.extend(img_score.cpu().tolist())
            all_labels.extend(labels.tolist())

            for j in range(B):
                pmap = anomaly_map[j].unsqueeze(0).unsqueeze(0).float()
                pmap = F.interpolate(pmap, size=(512, 512), mode='bilinear', align_corners=False)
                pixel_scores_list.append(pmap.squeeze().cpu().numpy())

        return np.array(image_scores), pixel_scores_list, np.array(all_labels)


# ============================================================
# EfficientAD (simplified teacher-student feature matching)
# (Batzner et al., WACV 2024)
# ============================================================

class EfficientADStudent(nn.Module):
    """Lightweight student for feature distillation."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class EfficientAD:
    """
    EfficientAD: Lightweight teacher-student anomaly detection.
    Teacher: frozen WideResNet-50. Student: small CNN trained via feature
    distillation on normal samples. Anomaly scored by feature discrepancy.

    Reference: Batzner et al., WACV 2024
    """

    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.teacher = WideResNet50FeatureExtractor(device=self.device).to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.student = EfficientADStudent().to(self.device)

    def fit(self, train_loader, epochs=60, lr=1e-3):
        """Train student to match teacher features."""
        self.teacher.eval()
        self.student.train()
        optimizer = optim.AdamW(self.student.parameters(), lr=lr, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print(f"EfficientAD: training student ({epochs} epochs, lr={lr})")
        pbar = tqdm(range(epochs), desc='EfficientAD training')
        best_loss = float('inf')

        for epoch in pbar:
            total_loss = 0.0
            for batch in train_loader:
                images = batch['image'].to(self.device)
                with torch.no_grad():
                    teacher_feats = self.teacher(images)
                    # Use layer2 output, pool to match student spatial size
                    t_feat = F.adaptive_avg_pool2d(teacher_feats[0], (16, 16))

                s_feat = self.student(images)
                # Feature distillation loss (cosine + MSE)
                loss = F.mse_loss(s_feat, t_feat) + \
                       0.3 * (1 - F.cosine_similarity(
                           s_feat.flatten(1), t_feat.flatten(1), dim=1)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'best': f'{best_loss:.4f}'})

        return self

    @torch.no_grad()
    def predict(self, loader):
        """Feature discrepancy as anomaly score."""
        self.teacher.eval()
        self.student.eval()
        image_scores = []
        pixel_scores_list = []
        all_labels = []

        for batch in tqdm(loader, desc='EfficientAD: predicting'):
            images = batch['image'].to(self.device)
            labels = batch['label']
            teacher_feats = self.teacher(images)
            t_feat = F.adaptive_avg_pool2d(teacher_feats[0], (16, 16))
            s_feat = self.student(images)

            # Per-location cosine distance
            t_norm = F.normalize(t_feat.flatten(2), dim=1)  # (B, C, HW)
            s_norm = F.normalize(s_feat.flatten(2), dim=1)
            anomaly_map = 1 - (t_norm * s_norm).sum(dim=1)  # (B, HW)
            anomaly_map = anomaly_map.reshape(images.shape[0], 16, 16)

            img_score = anomaly_map.reshape(images.shape[0], -1).max(dim=1).values
            image_scores.extend(img_score.cpu().tolist())
            all_labels.extend(labels.tolist())

            for j in range(images.shape[0]):
                pmap = anomaly_map[j].unsqueeze(0).unsqueeze(0).float()
                pmap = F.interpolate(pmap, size=(512, 512), mode='bilinear', align_corners=False)
                pixel_scores_list.append(pmap.squeeze().cpu().numpy())

        return np.array(image_scores), pixel_scores_list, np.array(all_labels)


# ============================================================
# Utility functions
# ============================================================

def build_test_loader(config):
    """Build test data loader from config."""
    data_config = config.get('data', {})
    test_dataset = CSVDataset(
        csv_path=data_config.get('test_csv', 'data/test.csv'),
        images_dir=data_config.get('images_dir', 'data/images'),
        split='test',
        image_size=data_config.get('image_size', 512),
        augment=False,
    )
    return DataLoader(test_dataset, batch_size=8, shuffle=False,
                      num_workers=data_config.get('num_workers', 4), pin_memory=True)


def build_train_loader(config):
    """Build train data loader (normal-only for fitting)."""
    data_config = config.get('data', {})
    train_dataset = CSVDataset(
        csv_path=data_config.get('train_csv', 'data/train.csv'),
        images_dir=data_config.get('images_dir', 'data/images'),
        split='train',
        image_size=data_config.get('image_size', 512),
        augment=False,
    )
    return DataLoader(train_dataset, batch_size=16, shuffle=False,
                      num_workers=data_config.get('num_workers', 4), pin_memory=True)


def evaluate_method(image_scores, pixel_maps, labels, masks_list, method_name):
    """Compute standard evaluation metrics."""
    results = {}

    # Image-level
    results['I-AUROC'] = compute_auroc(image_scores, labels)
    results['I-F1max'] = compute_f1_max(image_scores, labels)[0]
    results['I-AU-PR'] = compute_auprc(image_scores, labels)

    # Pixel-level (if masks available)
    if masks_list and len(masks_list) > 0:
        px_scores = np.concatenate([p.flatten() for p in pixel_maps])
        px_labels = np.concatenate([m.flatten() for m in masks_list])
        results['P-AUROC'] = compute_auroc(px_scores, px_labels)

    print(f"\n{'='*60}")
    print(f"  {method_name} Results")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:>12s}: {v:.4f}")
    print(f"{'='*60}\n")

    return results
