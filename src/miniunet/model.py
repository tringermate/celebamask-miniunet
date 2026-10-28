"""Mini-U-Net for 19-class face parsing under a 1.8M-parameter budget.

With every ``use_*`` switch at its default, the network has exactly 1,818,398
trainable parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise convolution followed by a 1x1 pointwise projection (no bias)."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module: channel attention, then spatial attention.

    Args:
        c: Number of input (and output) channels.
        r: Reduction ratio of the channel-attention MLP.
    """

    def __init__(self, c: int, r: int = 16) -> None:
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Linear(c, max(1, c // r)),
            nn.ReLU(True),
            nn.Linear(max(1, c // r), c),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ch, _, _ = x.size()
        ca = self.ex(self.sq(x).view(b, ch)).view(b, ch, 1, 1)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x * sa


class ResidualLightweightConv(nn.Module):
    """Two DSConv-BN-ReLU layers whose outputs are summed, then passed through CBAM.

    Args:
        in_c: Input channels.
        out_c: Output channels.
        use_cbam: If False, CBAM is replaced by an identity (ablation only).
        cbam_reduction: Reduction ratio passed to ``CBAMBlock``.
    """

    def __init__(
        self, in_c: int, out_c: int, use_cbam: bool = True, cbam_reduction: int = 16
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(in_c, out_c), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(out_c, out_c), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
        self.cbam = CBAMBlock(out_c, cbam_reduction) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return self.cbam(x1 + x2)


class DilatedContextBlock(nn.Module):
    """Sum of the input and three depthwise 3x3 convolutions (dilation 1, 2, 4), then BN-ReLU."""

    def __init__(self, c: int = 512) -> None:
        super().__init__()
        self.d1 = nn.Conv2d(c, c, 3, padding=1, dilation=1, groups=c, bias=False)
        self.d2 = nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c, bias=False)
        self.d4 = nn.Conv2d(c, c, 3, padding=4, dilation=4, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(x + self.d1(x) + self.d2(x) + self.d4(x)))


class AttentionGate(nn.Module):
    """Additive attention gate: scales skip features ``x`` by a mask computed from ``g`` and ``x``.

    Args:
        F_g: Channels of the gating (decoder) signal.
        F_l: Channels of the skip (encoder) features.
        F_int: Channels of the intermediate projection.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = torch.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


class MiniUNet(nn.Module):
    """Four-level U-Net with lightweight residual blocks, CBAM, dilated context and attention gates.

    Args:
        n_channels: Input image channels.
        n_classes: Number of segmentation classes.
        dropout: Dropout2d probability at the bottleneck.
        cbam_reduction: CBAM channel reduction ratio.
        use_cbam: Ablation switch; False replaces every CBAM with an identity.
        use_attention_gates: Ablation switch; False passes skip features through unchanged.
        use_dilated_context: Ablation switch; False replaces the context block with an identity.

    Turning any switch off removes parameters, so the count no longer equals 1,818,398.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 19,
        dropout: float = 0.1,
        cbam_reduction: int = 16,
        use_cbam: bool = True,
        use_attention_gates: bool = True,
        use_dilated_context: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention_gates = use_attention_gates

        def block(in_c: int, out_c: int) -> ResidualLightweightConv:
            return ResidualLightweightConv(in_c, out_c, use_cbam, cbam_reduction)

        self.inc = block(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), block(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), block(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), block(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), block(256, 512))

        self.context = DilatedContextBlock(512) if use_dilated_context else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)

        if use_attention_gates:
            self.ag1 = AttentionGate(F_g=256, F_l=256, F_int=128)
            self.ag2 = AttentionGate(F_g=128, F_l=128, F_int=64)
            self.ag3 = AttentionGate(F_g=64, F_l=64, F_int=32)
            self.ag4 = AttentionGate(F_g=32, F_l=32, F_int=16)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv_up1 = block(512, 256)
        self.refinement = block(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv_up2 = block(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv_up3 = block(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv_up4 = block(64, 32)

        self.outc = nn.Conv2d(32, n_classes, 1)

    def _skip(self, gate: str, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the named attention gate, or return ``x`` unchanged when gates are off."""
        if not self.use_attention_gates:
            return x
        return getattr(self, gate)(g=g, x=x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.down4(x4)
        x5 = self.dropout(self.context(x5))

        g1 = self.up1(x5)
        x = self.conv_up1(torch.cat([g1, self._skip("ag1", g1, x4)], 1))
        x = self.refinement(x)

        g2 = self.up2(x)
        x = self.conv_up2(torch.cat([g2, self._skip("ag2", g2, x3)], 1))

        g3 = self.up3(x)
        x = self.conv_up3(torch.cat([g3, self._skip("ag3", g3, x2)], 1))

        g4 = self.up4(x)
        x = self.conv_up4(torch.cat([g4, self._skip("ag4", g4, x1)], 1))

        return self.outc(x)


def build_model(model_cfg: dict) -> MiniUNet:
    """Build a ``MiniUNet`` from the ``model`` section of the config."""
    return MiniUNet(
        n_channels=3,
        n_classes=model_cfg.get("n_classes", 19),
        dropout=model_cfg.get("dropout", 0.1),
        cbam_reduction=model_cfg.get("cbam_reduction", 16),
        use_cbam=model_cfg.get("use_cbam", True),
        use_attention_gates=model_cfg.get("use_attention_gates", True),
        use_dilated_context=model_cfg.get("use_dilated_context", True),
    )


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
