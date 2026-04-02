import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import math
import shutil
from pathlib import Path

import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from datasets.polyp_datasets import build_train_val_loaders, set_seed
from losses.mesa_loss import MESALoss
from model.mesa_net import build_mesa_net
from utils.metrics import dice_score, iou_score, recall_score


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(device_name):
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        return 'cpu'
    return device_name


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    dice_values, iou_values, recall_values = [], [], []
    for images, masks in tqdm(loader, desc='Validation', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images, return_aux=False)
        pred = (torch.sigmoid(logits) > threshold).float()
        for pred_i, mask_i in zip(pred, masks):
            dice_values.append(dice_score(pred_i, mask_i))
            iou_values.append(iou_score(pred_i, mask_i))
            recall_values.append(recall_score(pred_i, mask_i))
    return {
        'dice': float(sum(dice_values) / max(1, len(dice_values))),
        'iou': float(sum(iou_values) / max(1, len(iou_values))),
        'recall': float(sum(recall_values) / max(1, len(recall_values))),
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running = []
    for images, masks in tqdm(loader, desc='Train', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images, return_aux=True)
            loss, _ = criterion(outputs, masks)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        running.append(float(loss.item()))
    return float(sum(running) / max(1, len(running)))


def make_scheduler(optimizer, total_epochs, base_lr):
    def lr_lambda(epoch):
        warmup_epochs = max(1, int(total_epochs * 0.1))
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        eta_min = 5e-7 / base_lr
        return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser(description='Train MESA-Net on Kvasir-SEG.')
    parser.add_argument('--config', type=str, default='configs/mesa_net.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg['train']
    loss_cfg = cfg['loss']
    eval_cfg = cfg.get('eval', {})

    set_seed(int(train_cfg.get('seed', 42)))
    device = resolve_device(str(cfg.get('device', 'cuda')))

    train_loader, val_loader = build_train_val_loaders(cfg)
    model = build_mesa_net(num_classes=1, use_aligned_auxiliary_heads=True).to(device)
    criterion = MESALoss(
        aux_weights=tuple(loss_cfg.get('aux_weights', [0.3, 0.15])),
        boundary_weight=float(loss_cfg.get('boundary_weight', 0.05)),
        distill_weight=float(loss_cfg.get('distill_weight', 0.1)),
        temperature=float(loss_cfg.get('temperature', 4.0)),
    )
    optimizer = Adam(model.parameters(), lr=float(train_cfg.get('lr', 1e-4)), weight_decay=float(train_cfg.get('weight_decay', 0.0)))
    scheduler = make_scheduler(optimizer, int(train_cfg.get('epochs', 200)), float(train_cfg.get('lr', 1e-4)))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get('use_amp', True)) and device.startswith('cuda'))
    if not scaler.is_enabled():
        scaler = None

    exp_name = str(cfg.get('experiment_name', 'mesa_net_run'))
    output_root = Path(train_cfg.get('save_dir', 'outputs')) / exp_name
    ckpt_dir = output_root / 'checkpoints'
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_root / 'used_config.yaml')

    threshold = float(eval_cfg.get('threshold', 0.5))
    epochs = int(train_cfg.get('epochs', 200))
    best_dice = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, device, threshold=threshold)
        scheduler.step()

        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_dice': val_metrics['dice'],
            'val_iou': val_metrics['iou'],
            'val_recall': val_metrics['recall'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': cfg,
            'metrics': record,
        }
        torch.save(state, ckpt_dir / 'last.pth')
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(state, ckpt_dir / 'best.pth')

    with open(output_root / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f'Finished. Best validation Dice: {best_dice:.4f}')


if __name__ == '__main__':
    main()
