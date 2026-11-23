r"""Part II, Section 8.2: the network.

A U-Net (Ronneberger et al. 2015): an encoder of ``depth`` levels, each two
:math:`3\times 3` convolutions (group normalisation, SiLU) followed by a halving of
the resolution and a doubling of the channels, the mirrored decoder with skip
connections, and a :math:`1\times 1` convolution to one output channel, the heat map
of Eq. (8.1).  The input is ``in_channels`` standardised images (target and
second target).  The sides of the input must be multiples of :math:`2^{\rm depth}`;
:func:`pad_to` and :func:`predict` take care of that.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _block(c_in: int, c_out: int) -> nn.Sequential:
    groups = 8 if c_out % 8 == 0 else 1
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(groups, c_out), nn.SiLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(groups, c_out), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, in_channels: int = 2, base: int = 32, depth: int = 4):
        super().__init__()
        self.config: Dict[str, int] = {"in_channels": int(in_channels), "base": int(base), "depth": int(depth)}
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(in_channels, widths[0])]
                                  + [_block(widths[i], widths[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(depth))])
        self.merge = nn.ModuleList([_block(2 * widths[i], widths[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(widths[0], 1, 1)

    @property
    def multiple(self) -> int:
        return 2 ** self.config["depth"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()
        for up, merge in zip(self.up, self.merge):
            x = merge(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)[:, 0]


def pad_to(x: torch.Tensor, multiple: int):
    """Reflect-pad the last two sides of ``x`` up to multiples of ``multiple``; returns ``(padded, (rows, cols))``."""

    rows, cols = x.shape[-2:]
    pr, pc = (-rows) % multiple, (-cols) % multiple
    if pr or pc:
        mode = "reflect" if pr < rows and pc < cols else "replicate"
        x = F.pad(x, (0, pc, 0, pr), mode=mode)
    return x, (rows, cols)


@torch.no_grad()
def predict(net: UNet, image: np.ndarray, device, tile: int = 512, overlap: int = 32) -> np.ndarray:
    """Heat map of ``image`` ``(C, H, W)``; large images in overlapping tiles (only the inner part of a tile counts)."""

    net.eval()
    x = torch.as_tensor(np.asarray(image, dtype=np.float32), device=device)[None]
    _, _, rows, cols = x.shape
    step = max(tile - 2 * overlap, 16)
    if rows <= tile and cols <= tile:
        padded, _ = pad_to(x, net.multiple)
        return net(padded)[0, :rows, :cols].float().cpu().numpy()
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(0, rows, step):
        for c in range(0, cols, step):
            r0, c0 = max(r - overlap, 0), max(c - overlap, 0)
            r1, c1 = min(r + step + overlap, rows), min(c + step + overlap, cols)
            padded, _ = pad_to(x[..., r0:r1, c0:c1], net.multiple)
            heat = net(padded)[0, :r1 - r0, :c1 - c0].float().cpu().numpy()
            rr, cc = min(r + step, rows), min(c + step, cols)
            out[r:rr, c:cc] = heat[r - r0:rr - r0, c - c0:cc - c0]
    return out
