import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.polyp_datasets import build_train_val_loaders, set_seed
from losses.mesa_loss import MESALoss
from model.mesa_net import build_mesa_net
from utils.metrics import dice_score, iou_score, recall_score


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_name: str) -> str:
    device_name = str(device_name).lower()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Falling back to CPU.")
        return "cpu"
    return device_name


def move_tensor(x: torch.Tensor, device: str) -> torch.Tensor:
    non_blocking = device.startswith("cuda")
    return x.to(device, non_blocking=non_blocking)


@torch.no_grad()
def evaluate(model, loader, device: str, threshold: float = 0.5):
    model.eval()
    dice_values, iou_values, recall_values = [], [], []

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = move_tensor(images, device)
        masks = move_tensor(masks, device)

        logits = model(images, return_aux=False)
        pred = (torch.sigmoid(logits) > threshold).float()

        for pred_i, mask_i in zip(pred, masks):
            dice_values.append(dice_score(pred_i, mask_i))
            iou_values.append(iou_score(pred_i, mask_i))
            recall_values.append(recall_score(pred_i, mask_i))

    return {
        "dice": float(sum(dice_values) / max(1, len(dice_values))),
        "iou": float(sum(iou_values) / max(1, len(iou_values))),
        "recall": float(sum(recall_values) / max(1, len(recall_values))),
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device: str, amp_enabled: bool):
    model.train()
    running_losses = []

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = move_tensor(images, device)
        masks = move_tensor(masks, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
            outputs = model(images, return_aux=True)
            loss, _ = criterion(outputs, masks)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_losses.append(float(loss.item()))

    return float(sum(running_losses) / max(1, len(running_losses)))


def make_scheduler(optimizer, total_epochs: int, base_lr: float):
    def lr_lambda(epoch: int):
        warmup_epochs = max(1, int(total_epochs * 0.1))
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        eta_min_ratio = 5e-7 / base_lr
        return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train MESA-Net on Kvasir-SEG.")
    parser.add_argument("--config", type=str, default="configs/mesa_net.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    eval_cfg = cfg.get("eval", {})

    set_seed(int(train_cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cuda"))

    train_loader, val_loader = build_train_val_loaders(cfg)

    model = build_mesa_net(
        num_classes=1,
        use_aligned_auxiliary_heads=True,
    ).to(device)

    criterion = MESALoss(
        aux_weights=tuple(loss_cfg.get("aux_weights", [0.3, 0.15])),
        boundary_weight=float(loss_cfg.get("boundary_weight", 0.05)),
        distill_weight=float(loss_cfg.get("distill_weight", 0.1)),
        temperature=float(loss_cfg.get("temperature", 4.0)),
    )

    base_lr = float(train_cfg.get("lr", 1e-4))
    epochs = int(train_cfg.get("epochs", 200))

    optimizer = Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = make_scheduler(optimizer, epochs, base_lr)

    amp_enabled = bool(train_cfg.get("use_amp", True)) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    exp_name = str(cfg.get("experiment_name", "mesa_net_run"))
    output_root = Path(train_cfg.get("save_dir", "outputs")) / exp_name
    ckpt_dir = output_root / "checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, output_root / "used_config.yaml")

    threshold = float(eval_cfg.get("threshold", 0.5))
    best_dice = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            threshold=threshold,
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_recall": val_metrics["recall"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "metrics": record,
        }

        torch.save(state, ckpt_dir / "last.pth")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(state, ckpt_dir / "best.pth")

    with open(output_root / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"Finished. Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()