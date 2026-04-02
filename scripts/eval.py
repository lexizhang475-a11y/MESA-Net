import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import torch
import yaml
from tqdm import tqdm

from datasets.polyp_datasets import build_eval_loader
from model.mesa_net import build_mesa_net
from utils.metrics import dice_score, iou_score, recall_score


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold=0.5):
    model.eval()
    dice_values, iou_values, recall_values = [], [], []
    for batch in tqdm(loader, desc='Evaluate', leave=False):
        if len(batch) == 3:
            images, masks, _ = batch
        else:
            images, masks, _, _ = batch
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate MESA-Net on ClinicDB / ColonDB / PolypGen.')
    parser.add_argument('--config', type=str, default='configs/mesa_net.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='all', choices=['clinicdb', 'colondb', 'polypgen', 'all'])
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--use-degraded-input', action='store_true', help='Apply degradation to ClinicDB/ColonDB evaluation inputs. PolypGen remains clean.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    threshold = float(args.threshold if args.threshold is not None else cfg.get('eval', {}).get('threshold', 0.5))
    device = str(cfg.get('device', 'cuda'))
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    model = build_mesa_net(num_classes=1, use_aligned_auxiliary_heads=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    datasets = ['clinicdb', 'colondb', 'polypgen'] if args.dataset == 'all' else [args.dataset]
    results = {}
    for dataset_name in datasets:
        loader = build_eval_loader(
            cfg,
            dataset_name=dataset_name,
            batch_size=1,
            use_degraded_input=(args.use_degraded_input and dataset_name in {'clinicdb', 'colondb'})
        )
        results[dataset_name] = evaluate_loader(model, loader, device, threshold=threshold)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
