import torch
import torch.nn as nn
import torch.nn.functional as F


class MESALoss(nn.Module):
    """
    MESA loss from the paper:
      L = L_seg(main) + sum_k w_k * L_seg(aux_k) + lambda_b * L_edge + lambda_d * L_SD
    with:
      L_seg = BCEWithLogits + global Dice
      L_edge = Sobel-based boundary consistency
      L_SD   = BCE-based self-distillation
    """

    def __init__(
        self,
        aux_weights=(0.3, 0.15),
        boundary_weight=0.05,
        distill_weight=0.1,
        temperature=4.0,
        use_boundary_loss=True,
        use_self_distillation=True,
    ):
        super().__init__()
        self.aux_weights = list(aux_weights)
        self.boundary_weight = float(boundary_weight)
        self.distill_weight = float(distill_weight)
        self.temperature = float(temperature)
        self.use_boundary_loss = bool(use_boundary_loss)
        self.use_self_distillation = bool(use_self_distillation)
        self.bce = nn.BCEWithLogitsLoss()
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def global_dice_loss(self, logits, target, smooth=1.0):
        prob = torch.sigmoid(logits)
        prob = prob.reshape(-1)
        target = target.reshape(-1)
        intersection = (prob * target).sum()
        union = prob.sum() + target.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice

    def segmentation_loss(self, logits, target):
        return self.bce(logits, target) + self.global_dice_loss(logits, target)

    def boundary_response(self, logits):
        prob = torch.sigmoid(logits)
        gx = F.conv2d(prob, self.sobel_x.to(prob.dtype), padding=1)
        gy = F.conv2d(prob, self.sobel_y.to(prob.dtype), padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    def boundary_consistency_loss(self, main_logits, aux_logits_list):
        main_boundary = self.boundary_response(main_logits)
        losses = []
        for aux_logits in aux_logits_list:
            if aux_logits.shape[-2:] != main_logits.shape[-2:]:
                aux_logits = F.interpolate(aux_logits, size=main_logits.shape[-2:], mode='bilinear', align_corners=True)
            aux_boundary = self.boundary_response(aux_logits)
            losses.append(F.l1_loss(aux_boundary, main_boundary))
        return sum(losses) / max(1, len(losses))

    def self_distillation_loss(self, main_logits, aux_logits_list):
        teacher_prob = torch.sigmoid(main_logits / self.temperature).detach()
        losses = []
        for aux_logits in aux_logits_list:
            if aux_logits.shape[-2:] != main_logits.shape[-2:]:
                aux_logits = F.interpolate(aux_logits, size=main_logits.shape[-2:], mode='bilinear', align_corners=True)
            student_prob = torch.sigmoid(aux_logits / self.temperature)
            losses.append(F.binary_cross_entropy(student_prob, teacher_prob) * (self.temperature ** 2))
        return sum(losses) / max(1, len(losses))

    def forward(self, outputs, target):
        if isinstance(outputs, torch.Tensor):
            outputs = {'logits': outputs, 'aux': None}

        logits = outputs['logits']
        aux_logits = outputs.get('aux', None)
        total = self.segmentation_loss(logits, target)
        loss_dict = {
            'main': float(total.item()),
        }

        if aux_logits is not None:
            aux_total = 0.0
            for i, aux in enumerate(aux_logits):
                w = self.aux_weights[i] if i < len(self.aux_weights) else self.aux_weights[-1]
                aux_total = aux_total + w * self.segmentation_loss(aux, target)
            total = total + aux_total
            loss_dict['aux'] = float(aux_total.item())

            if self.use_boundary_loss:
                boundary = self.boundary_consistency_loss(logits, aux_logits)
                total = total + self.boundary_weight * boundary
                loss_dict['boundary'] = float(boundary.item())

            if self.use_self_distillation:
                distill = self.self_distillation_loss(logits, aux_logits)
                total = total + self.distill_weight * distill
                loss_dict['distill'] = float(distill.item())

        loss_dict['total'] = float(total.item())
        return total, loss_dict
