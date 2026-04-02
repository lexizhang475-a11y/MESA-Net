import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

from pathlib import Path
import random
import numpy as np
from PIL import Image
import albumentations as A
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils.random_lightspot import AddLightSpots


IMG_SIZE = 224
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _list_images(folder):
    folder = Path(folder)
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def _pair_by_stem(images_dir, masks_dir):
    image_files = _list_images(images_dir)
    mask_files = _list_images(masks_dir)
    mask_map = {p.stem: p for p in mask_files}
    pairs = []
    missing = []
    for img in image_files:
        mask = mask_map.get(img.stem)
        if mask is None:
            missing.append(img.name)
            continue
        pairs.append((str(img), str(mask)))
    if not pairs:
        raise RuntimeError(f'No paired image/mask files found in {images_dir} and {masks_dir}.')
    if missing:
        print(f'[WARN] Skipped {len(missing)} images without masks in {images_dir}.')
    return pairs


def build_train_augmentation(image_size=IMG_SIZE):
    return A.Compose([
        A.Resize(height=image_size, width=image_size, p=1.0),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Affine(scale=(0.9, 1.1), p=0.5),
    ], p=1.0)


def build_eval_augmentation(image_size=IMG_SIZE):
    return A.Compose([
        A.Resize(height=image_size, width=image_size, p=1.0),
    ], p=1.0)


def build_degradation_augmentation():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2, 0.2), p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 21), p=1.0),
        ], p=0.7),
        A.OpticalDistortion(distort_limit=0.05, p=0.2),
        AddLightSpots(radius_range=(5, 40), intensity=0.85, num_spots=1, p=0.6),
    ], p=1.0)


class PairedPolypDataset(Dataset):
    def __init__(self, pairs, image_transform=None, degradation_transform=None, return_meta=False):
        self.pairs = list(pairs)
        self.image_transform = image_transform
        self.degradation_transform = degradation_transform
        self.return_meta = return_meta

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        if self.image_transform is not None:
            transformed = self.image_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        if self.degradation_transform is not None:
            image = self.degradation_transform(image=image)['image']
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        if self.return_meta:
            return image, mask, Path(image_path).name
        return image, mask


class PolypGenDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, degradation_transform=None, return_meta=True):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f'PolypGen root not found: {root_dir}')
        self.image_transform = image_transform
        self.degradation_transform = degradation_transform
        self.return_meta = return_meta
        self.items = []
        seq_dirs = sorted([p for p in self.root_dir.iterdir() if p.is_dir() and p.name.startswith('seq')])
        for seq_dir in seq_dirs:
            image_dir = seq_dir / 'images'
            mask_dir = seq_dir / 'masks'
            if not image_dir.exists() or not mask_dir.exists():
                continue
            mask_map = {p.name: p for p in _list_images(mask_dir)}
            for img_path in _list_images(image_dir):
                mask_path = mask_map.get(img_path.name)
                if mask_path is None:
                    continue
                self.items.append((img_path, mask_path, seq_dir.name, img_path.stem))
        if not self.items:
            raise RuntimeError(f'No valid PolypGen samples found under {root_dir}.')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, mask_path, sequence_name, frame_id = self.items[index]
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        if self.image_transform is not None:
            transformed = self.image_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        if self.degradation_transform is not None:
            image = self.degradation_transform(image=image)['image']
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        if self.return_meta:
            return image, mask, sequence_name, frame_id
        return image, mask


def build_train_val_loaders(cfg):
    data_cfg = cfg['data']
    train_cfg = cfg['train']
    seed = int(train_cfg.get('seed', 42))
    val_ratio = float(train_cfg.get('val_ratio', 0.2))
    batch_size = int(train_cfg.get('batch_size', 16))
    num_workers = int(train_cfg.get('num_workers', 4))

    set_seed(seed)
    pairs = _pair_by_stem(data_cfg['kvasir_images_dir'], data_cfg['kvasir_masks_dir'])
    train_pairs, val_pairs = train_test_split(pairs, test_size=val_ratio, random_state=seed, shuffle=True)

    train_set = PairedPolypDataset(
        train_pairs,
        image_transform=build_train_augmentation(),
        degradation_transform=build_degradation_augmentation(),
        return_meta=False,
    )
    val_set = PairedPolypDataset(
        val_pairs,
        image_transform=build_eval_augmentation(),
        degradation_transform=None,
        return_meta=False,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_eval_loader(cfg, dataset_name, batch_size=1, use_degraded_input=False):
    data_cfg = cfg['data']
    num_workers = int(cfg.get('eval', {}).get('num_workers', 2))
    image_transform = build_eval_augmentation()
    degradation_transform = build_degradation_augmentation() if use_degraded_input else None

    dataset_name = dataset_name.lower()
    if dataset_name == 'clinicdb':
        dataset = PairedPolypDataset(
            _pair_by_stem(data_cfg['clinicdb_images_dir'], data_cfg['clinicdb_masks_dir']),
            image_transform=image_transform,
            degradation_transform=degradation_transform,
            return_meta=True,
        )
    elif dataset_name == 'colondb':
        dataset = PairedPolypDataset(
            _pair_by_stem(data_cfg['colondb_images_dir'], data_cfg['colondb_masks_dir']),
            image_transform=image_transform,
            degradation_transform=degradation_transform,
            return_meta=True,
        )
    elif dataset_name == 'polypgen':
        dataset = PolypGenDataset(
            root_dir=data_cfg['polypgen_root_dir'],
            image_transform=image_transform,
            degradation_transform=None,
            return_meta=True,
        )
    else:
        raise ValueError("dataset_name must be one of: clinicdb, colondb, polypgen")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
