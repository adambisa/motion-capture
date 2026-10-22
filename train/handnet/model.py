"""HandNet: a tiny CenterNet-style hand detector, written out by hand.

One RGB frame in, one tensor out at 1/4 resolution with 5 channels:
  0      hand-centre heatmap logit
  1, 2   log(box width / 4), log(box height / 4), in input pixels
  3, 4   sub-cell offset of the box centre (0..1)

The backbone is MobileNetV2-style (inverted residuals: 1x1 expand, 3x3
depthwise, 1x1 project), the neck a light top-down FPN back to stride 4.
Only ops OpenCV's DNN module runs well are used: Conv, BatchNorm, ReLU6,
Add, nearest Resize, Concat. `width` scales every layer; it and the input
size are the two speed/accuracy dials.
"""

import torch
from torch import nn

STRIDE = 4
OUT_CHANNELS = 5
HEAT_PRIOR = 0.1  # initial heatmap probability, so early focal loss is sane


def _ch(channels: int, width: float) -> int:
    """Channels scaled by width, rounded to a multiple of 8 (at least 8)."""
    return max(8, int(channels * width + 4) // 8 * 8)


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 1) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU6(inplace=True),
        )


class SepConv(nn.Sequential):
    """Depthwise 3x3 then pointwise 1x1: a cheap 3x3 convolution."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(ConvBNReLU(cin, cin, 3, groups=cin), ConvBNReLU(cin, cout, 1))


class InvertedResidual(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, expand: int) -> None:
        super().__init__()
        hidden = cin * expand
        layers: list[nn.Module] = []
        if expand != 1:
            layers.append(ConvBNReLU(cin, hidden, 1))
        layers += [
            ConvBNReLU(hidden, hidden, 3, stride, groups=hidden),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        ]
        self.block = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.residual else y


def _stage(cin: int, cout: int, blocks: int, stride: int, expand: int) -> nn.Sequential:
    layers = [InvertedResidual(cin, cout, stride, expand)]
    layers += [InvertedResidual(cout, cout, 1, expand) for _ in range(blocks - 1)]
    return nn.Sequential(*layers)


class HandNet(nn.Module):
    def __init__(self, width: float = 1.0) -> None:
        super().__init__()
        c = [_ch(v, width) for v in (16, 24, 32, 64, 96)]
        neck = _ch(32, width)
        self.stem = ConvBNReLU(3, c[0], 3, 2)  # 1/2
        self.s1 = _stage(c[0], c[1], 1, 2, 3)  # 1/4
        self.s2 = _stage(c[1], c[2], 3, 2, 4)  # 1/8
        self.s3 = _stage(c[2], c[3], 4, 2, 4)  # 1/16
        self.s4 = _stage(c[3], c[4], 2, 2, 4)  # 1/32
        self.lateral = nn.ModuleList(ConvBNReLU(ci, neck, 1) for ci in c[1:])
        self.smooth = nn.ModuleList(SepConv(neck, neck) for _ in range(3))
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Sequential(SepConv(neck, neck), nn.Conv2d(neck, OUT_CHANNELS, 1))
        bias = self.head[-1].bias
        assert bias is not None
        nn.init.zeros_(bias)
        nn.init.constant_(bias[:1], -torch.log(torch.tensor((1 - HEAT_PRIOR) / HEAT_PRIOR)).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.s1(self.stem(x))
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        p = self.lateral[3](f4)
        for i, f in ((2, f3), (1, f2), (0, f1)):  # top-down, back to stride 4
            p = self.smooth[i](self.up(p) + self.lateral[i](f))
        return self.head(p)


def complexity(model: nn.Module, height: int, width: int) -> tuple[int, int]:
    """(parameters, multiply-accumulates) for one input, counted with forward hooks."""
    macs = 0

    def hook(module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1]
            macs += out.numel() * k * module.in_channels // module.groups

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, height, width))
    for h in handles:
        h.remove()
    return sum(p.numel() for p in model.parameters()), macs
