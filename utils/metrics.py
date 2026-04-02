import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def dice_score(pred, target, eps=1e-6):
    pred = _to_numpy(pred).astype(np.float32)
    target = _to_numpy(target).astype(np.float32)
    intersection = (pred * target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * intersection + eps) / (denom + eps))


def iou_score(pred, target, eps=1e-6):
    pred = _to_numpy(pred).astype(np.float32)
    target = _to_numpy(target).astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def recall_score(pred, target, eps=1e-6):
    pred = _to_numpy(pred).astype(np.float32)
    target = _to_numpy(target).astype(np.float32)
    tp = (pred * target).sum()
    fn = ((1.0 - pred) * target).sum()
    denom = tp + fn
    if denom == 0:
        return 1.0
    return float((tp + eps) / (denom + eps))
