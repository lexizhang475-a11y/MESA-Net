import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvolutionalEncoder(nn.Module):
    """Purely convolutional encoder used in MESA-Net."""

    def __init__(self):
        super().__init__()
        self.stage1_1 = ConvBNReLU(3, 32, kernel_size=7, stride=2, padding=3)
        self.stage1_2 = ConvBNReLU(32, 64, kernel_size=3, stride=2, padding=1)   # F4
        self.stage2_1 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.stage2_2 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.stage2_3 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)   # F8
        self.stage3_1 = ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage3_2 = ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
        self.stage3_3 = ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1) # F16

    def forward(self, x):
        x = self.stage1_1(x)
        f4 = self.stage1_2(x)
        x = self.stage2_1(f4)
        x = self.stage2_2(x)
        f8 = self.stage2_3(x)
        x = self.stage3_1(f8)
        _ = self.stage3_2(x)
        f16 = self.stage3_3(_)
        return f4, f8, f16


class AxialDepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_size=(3, 3), dilation=1):
        super().__init__()
        kh, kw = kernel_size
        self.dw_h = nn.Conv2d(channels, channels, kernel_size=(kh, 1), padding=(dilation, 0), groups=channels, dilation=dilation)
        self.dw_w = nn.Conv2d(channels, channels, kernel_size=(1, kw), padding=(0, dilation), groups=channels, dilation=dilation)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)


class GhostAxialBlock(nn.Module):
    """GAB in the paper."""

    def __init__(self, channels):
        super().__init__()
        ghost_channels = channels // 4
        self.reduce = nn.Conv2d(channels, ghost_channels, kernel_size=1, bias=True)
        self.branch_d1 = AxialDepthwiseConv(ghost_channels, kernel_size=(3, 3), dilation=1)
        self.branch_d2 = AxialDepthwiseConv(ghost_channels, kernel_size=(3, 3), dilation=2)
        self.branch_d3 = AxialDepthwiseConv(ghost_channels, kernel_size=(3, 3), dilation=3)
        self.bn = nn.BatchNorm2d(ghost_channels * 4)
        self.project = nn.Conv2d(ghost_channels * 4, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        reduced = self.reduce(x)
        merged = torch.cat([
            reduced,
            self.branch_d1(reduced),
            self.branch_d2(reduced),
            self.branch_d3(reduced),
        ], dim=1)
        return self.act(self.project(self.bn(merged)))


class _AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fuse = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lateral, deep):
        deep_up = F.interpolate(deep, size=lateral.shape[2:], mode='nearest')
        x = torch.cat([lateral, deep_up], dim=1)
        x = self.fuse(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = self.conv(x)
        x = self.bn(x)
        return self.sigmoid(x)


class AttentionFusionModule(nn.Module):
    """AFM in the paper."""

    def __init__(self, deep_channels, lateral_channels, out_channels):
        super().__init__()
        self.lateral_proj = ConvBNReLU(lateral_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deep_proj = ConvBNReLU(deep_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.attention = _AttentionGate(in_channels=out_channels * 2, out_channels=out_channels)
        self.lateral_refine = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deep_refine = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, deep, lateral):
        lateral_feat = self.lateral_proj(lateral)
        deep_feat = self.deep_proj(deep)
        attention = self.attention(lateral_feat, deep_feat)
        lateral_feat = self.lateral_refine(lateral_feat) * (1.0 - attention)
        deep_feat = self.deep_refine(deep_feat) * attention
        deep_feat = F.interpolate(deep_feat, size=lateral_feat.shape[2:], mode='nearest')
        return self.smooth(lateral_feat + deep_feat)


class LiteAttentionFusionModule(nn.Module):
    """Lite-AFM in the paper."""

    def __init__(self, deep_channels, lateral_channels, out_channels, up_mode='nearest', align_corners=True):
        super().__init__()
        self.up_mode = up_mode
        self.align_corners = align_corners
        self.deep_proj = nn.Sequential(
            nn.Conv2d(deep_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.lateral_proj = nn.Sequential(
            nn.Conv2d(lateral_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, deep, lateral):
        if self.up_mode == 'bilinear':
            deep = F.interpolate(deep, size=lateral.shape[2:], mode='bilinear', align_corners=self.align_corners)
        else:
            deep = F.interpolate(deep, size=lateral.shape[2:], mode='nearest')
        deep = self.deep_proj(deep)
        lateral = self.lateral_proj(lateral)
        base = deep + lateral
        gate = self.gate(base)
        fused = lateral + gate * deep
        return self.refine(fused)


class DeepSupervisionAligner(nn.Module):
    """Aligner used to project auxiliary outputs into a common prediction space."""

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        hidden_channels = max(1, in_channels // 2)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise_bn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x, out_size):
        x = self.depthwise(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_out(x)
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=True)
        return x


class MESANet(nn.Module):
    """MESA-Net: encoder + GAB + AFM + Lite-AFM + auxiliary aligned heads."""

    def __init__(self, num_classes=1, use_aligned_auxiliary_heads=True, lite_up_mode='nearest'):
        super().__init__()
        self.use_aligned_auxiliary_heads = use_aligned_auxiliary_heads
        self.encoder = ConvolutionalEncoder()
        self.gab = GhostAxialBlock(channels=128)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.afm = AttentionFusionModule(deep_channels=128, lateral_channels=64, out_channels=64)
        self.lite_afm = LiteAttentionFusionModule(deep_channels=64, lateral_channels=64, out_channels=64, up_mode=lite_up_mode)
        self.main_head = nn.Conv2d(64, num_classes, kernel_size=1, bias=True)
        self.aligner_8 = DeepSupervisionAligner(in_channels=64, out_channels=num_classes)
        self.aligner_16 = DeepSupervisionAligner(in_channels=128, out_channels=num_classes)

    def forward(self, x, return_aux=None):
        if return_aux is None:
            return_aux = self.training and self.use_aligned_auxiliary_heads

        f4, f8, f16 = self.encoder(x)
        f16_enhanced = f16 + self.alpha * self.gab(f16)
        d8 = self.afm(f16_enhanced, f8)
        d4 = self.lite_afm(d8, f4)
        logits_4 = self.main_head(d4)
        logits = F.interpolate(logits_4, size=x.shape[2:], mode='bilinear', align_corners=True)

        if not return_aux:
            return logits

        aux_8 = self.aligner_8(d8, out_size=x.shape[2:])
        aux_16 = self.aligner_16(f16_enhanced, out_size=x.shape[2:])
        return {'logits': logits, 'aux': [aux_8, aux_16]}


def build_mesa_net(**kwargs):
    return MESANet(**kwargs)
