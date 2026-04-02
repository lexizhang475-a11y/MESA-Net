import torch
import torch.nn as nn
import torch.nn.functional as F


class MESALoss(nn.Module):
    """
    MESA Loss:
      L = Lseg(main) + sum_k w_k * Lseg(aux_k) + lambda_b * Ledge + lambda_d * LSD
    where:
      Lseg = BCEWithLogits + Global Dice
      Ledge = Sobel boundary consistency (L1)
      LSD  = BCE-based self-distillation with temperature T and factor T^2
    """

    def __init__(
        self,
        aux_weights=(0.3, 0.15),
        boundary_weight=0.05,
        distill_weight=0.1,
        temperature=4.0,
        use_boundary_loss=True,
        use_self_distill=True,
    ):
        super().__init__()
        self.aux_weights = list(aux_weights)
        self.boundary_weight = float(boundary_weight)
        self.distill_weight = float(distill_weight)
        self.temperature = float(temperature)
        self.use_boundary_loss = bool(use_boundary_loss)
        self.use_self_distill = bool(use_self_distill)

        self.bce = nn.BCEWithLogitsLoss()

        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32
            ).view(1, 1, 3, 3),
            persistent=False,
        )

    def dice_loss_global(self, logits, target, smooth=1.0):
        prob = torch.sigmoid(logits)
        prob = prob.reshape(-1)
        target = target.reshape(-1)
        inter = (prob * target).sum()
        union = prob.sum() + target.sum()
        dice = (2.0 * inter + smooth) / (union + smooth)
        return 1.0 - dice

    def boundary_response(self, logits):
        # 强制 float32，避免 AMP 下 half 与 Sobel kernel 不匹配
        prob = torch.sigmoid(logits).float()
        sobel_x = self.sobel_x.to(device=prob.device, dtype=prob.dtype)
        sobel_y = self.sobel_y.to(device=prob.device, dtype=prob.dtype)
        gx = F.conv2d(prob, sobel_x, padding=1)
        gy = F.conv2d(prob, sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def boundary_consistency_loss(self, main_logits, aux_logits_list):
        main_boundary = self.boundary_response(main_logits)
        losses = []
        for aux_logits in aux_logits_list:
            if aux_logits.shape[-2:] != main_logits.shape[-2:]:
                aux_logits = F.interpolate(
                    aux_logits,
                    size=main_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            aux_boundary = self.boundary_response(aux_logits)
            losses.append(F.l1_loss(aux_boundary, main_boundary))
        if len(losses) == 0:
            return main_logits.new_tensor(0.0)
        return sum(losses) / len(losses)

    def self_distillation_loss(self, main_logits, aux_logits_list):
        """
        BCE-based self-distillation:
          T^2 * BCE(sigmoid(student/T), sigmoid(teacher/T))

        注意：
        binary_cross_entropy 在 autocast 下不安全，所以这里显式关闭 autocast，
        并在 float32 中计算。
        """
        losses = []
        device_type = main_logits.device.type

        with torch.amp.autocast(device_type=device_type, enabled=False):
            teacher_prob = torch.sigmoid(main_logits.float() / self.temperature).detach()

            for aux_logits in aux_logits_list:
                if aux_logits.shape[-2:] != main_logits.shape[-2:]:
                    aux_logits = F.interpolate(
                        aux_logits,
                        size=main_logits.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )

                student_prob = torch.sigmoid(aux_logits.float() / self.temperature)
                loss = F.binary_cross_entropy(student_prob, teacher_prob) * (self.temperature ** 2)
                losses.append(loss)

        if len(losses) == 0:
            return main_logits.new_tensor(0.0)
        return sum(losses) / len(losses)

    def forward(self, outputs, targets):
        # 兼容两种输出格式：
        # 1) tensor logits
        # 2) {"logits": ..., "aux": [...]}
        if isinstance(outputs, torch.Tensor):
            logits = outputs
            aux_logits = []
        elif isinstance(outputs, dict):
            logits = outputs["logits"]
            aux_logits = outputs.get("aux", []) or []
        else:
            raise TypeError(f"Unsupported outputs type: {type(outputs)}")

        main_bce = self.bce(logits, targets)
        main_dice = self.dice_loss_global(logits, targets)
        main_loss = main_bce + main_dice

        total_loss = main_loss
        loss_dict = {
            "main_bce": float(main_bce.item()),
            "main_dice": float(main_dice.item()),
            "main": float(main_loss.item()),
        }

        if len(aux_logits) > 0:
            aux_loss = 0.0
            for i, aux in enumerate(aux_logits):
                if aux.shape[-2:] != targets.shape[-2:]:
                    aux = F.interpolate(
                        aux,
                        size=targets.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                w = self.aux_weights[i] if i < len(self.aux_weights) else self.aux_weights[-1]
                aux_bce = self.bce(aux, targets)
                aux_dice = self.dice_loss_global(aux, targets)
                aux_loss = aux_loss + w * (aux_bce + aux_dice)

            total_loss = total_loss + aux_loss
            loss_dict["aux"] = float(aux_loss.item())

            if self.use_boundary_loss:
                boundary = self.boundary_consistency_loss(logits, aux_logits)
                total_loss = total_loss + self.boundary_weight * boundary
                loss_dict["boundary"] = float(boundary.item())

            if self.use_self_distill:
                distill = self.self_distillation_loss(logits, aux_logits)
                total_loss = total_loss + self.distill_weight * distill
                loss_dict["distill"] = float(distill.item())

        loss_dict["total"] = float(total_loss.item())
        return total_loss, loss_dict