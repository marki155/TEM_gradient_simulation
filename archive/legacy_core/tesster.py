import numpy as np
import torch

def create_rho(pxpositions, intensities, multiplier=1.0):
    # pxpositions: (N, 2) Tensor
    # intensities: (N,) Tensor

    intensities = intensities.unsqueeze(1)  # (N, 1) Tensor

    pad = 2

    rho = torch.zeros((100 + pad * 2, 100 + pad * 2))

    # potential.fill_(self.background)  # Torch-Äquivalent

    py, px = pxpositions[:, 0], pxpositions[:, 1] # TODO warum vertauscht

    pos_mask = (px >= 0.0) & (px <= 100) & (py >= 0.0) & (py <= 100)
    px = px[pos_mask]
    py = py[pos_mask]
    intensities = intensities[pos_mask]
    # offset = int(self.canvas_size / self.kernelsize)
    offset = pad

    # Integer-Anteile
    # ix = torch.clamp(px.floor().to(torch.int64), min=0, max=self.shape[0] + self.canvas_size - 2)
    # iy = torch.clamp(py.floor().to(torch.int64), min=0, max=self.shape[1] + self.canvas_size - 2)

    ix = px.floor().to(torch.int64)
    iy = py.floor().to(torch.int64)

    # Bruchteile
    fx = px - ix
    fy = py - iy

    # Pixelgewichte
    w00 = torch.matmul((1 - fx), (1 - fy)) * intensities * multiplier
    w10 = torch.matmul(fx, (1 - fy)) * intensities* multiplier
    w11 = torch.matmul(fx, fy) * intensities* multiplier
    w01 = torch.matmul((1 - fx), fy) * intensities* multiplier

    # Indizes für alle vier Nachbarn
    idx00 = (offset + iy, offset + ix)
    idx10 = (offset + iy, offset + ix + 1)
    idx11 = (offset + iy + 1, offset + ix + 1)
    idx01 = (offset + iy + 1, offset + ix)

    # In-place Akkumulation wie np.add.at
    rho.index_put_(idx00, w00[:idx00[0].shape[0], 0], accumulate=True)
    rho.index_put_(idx10, w10[:idx10[0].shape[0], 0], accumulate=True)
    rho.index_put_(idx11, w11[:idx11[0].shape[0], 0], accumulate=True)
    rho.index_put_(idx01, w01[:idx01[0].shape[0], 0], accumulate=True)

    return torch.abs(rho[pad:-pad, pad:-pad])  # Entfernen des Pads


def create_rho2(positions_xy, intensities, multiplier=1.0):
    """
    positions_xy: (N, 2) mit [:,0]=x und [:,1]=y
    intensities:  (N,)
    """
    intensities = intensities.unsqueeze(1)
    pad = 2

    # Bildtensor immer als (H, W)
    rho = torch.zeros((100 + 2 * pad, 100 + 2 * pad))

    x = positions_xy[:, 0]
    y = positions_xy[:, 1]


    mask = (x >= 0.0) & (x < 100 - 1) & (y >= 0.0) & (y < 100 - 1)
    x = x[mask]
    y = y[mask]
    intensities = intensities[mask]

    ix = torch.floor(x).to(torch.int64)
    iy = torch.floor(y).to(torch.int64)

    fx = x - ix
    fy = y - iy

    # bilineare Gewichte
    w00 = ((1 - fx) * (1 - fy) * intensities[:, 0]) * multiplier
    w10 = (fx * (1 - fy) * intensities[:, 0]) * multiplier
    w01 = ((1 - fx) * fy * intensities[:, 0]) * multiplier
    w11 = (fx * fy * intensities[:, 0]) * multiplier





    idx00 = (pad + ix, pad + iy)
    idx10 = (pad + ix, pad + iy + 1)
    idx11 = (pad + ix + 1, pad + iy + 1)
    idx01 = (pad + ix + 1, pad + iy)

    # In-place Akkumulation wie np.add.at
    rho.index_put_(idx00, w00, accumulate=True)
    rho.index_put_(idx10, w10, accumulate=True)
    rho.index_put_(idx11, w11, accumulate=True)
    rho.index_put_(idx01, w01, accumulate=True)

    return torch.abs(rho[pad:-pad, pad:-pad])


N = 100
pxpositions = torch.zeros((N,2))
pxpositions = torch.tensor([np.linspace(0,100,101),np.linspace(0,100,101)]).transpose(0,1)
intensities = torch.ones((N+1))

create_rho(pxpositions, intensities)
create_rho2(pxpositions, intensities)


