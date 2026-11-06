"""Minimal CPU preview renderer: z-buffered splats of coloured vertices into a pinhole camera.

Good enough to compare a reconstruction against the input photos. Not a real rasteriser.
"""
import numpy as np
import torch


def render_points(verts: torch.Tensor, colors: torch.Tensor, K: torch.Tensor, E: torch.Tensor,
                  hw: tuple[int, int], radius: int = 1, bg: float = 1.0) -> np.ndarray:
    H, W = hw
    Xc = verts @ E[:3, :3].T + E[:3, 3]
    z = Xc[:, 2]
    ok = z > 1e-6
    u = (K[0, 0] * Xc[:, 0] / z + K[0, 2]).round().long()
    v = (K[1, 1] * Xc[:, 1] / z + K[1, 2]).round().long()
    z, u, v, c = z[ok].numpy(), u[ok].numpy(), v[ok].numpy(), colors[ok].numpy()
    zbuf = np.full((H, W), np.inf, np.float32)
    img = np.full((H, W, 3), bg, np.float32)
    order = np.argsort(-z)  # far to near, so nearer splats overwrite
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu, vv = u[order] + dx, v[order] + dy
            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            uu, vv, zz, cc = uu[inb], vv[inb], z[order][inb], c[order][inb]
            closer = zz < zbuf[vv, uu]
            zbuf[vv[closer], uu[closer]] = zz[closer]
            img[vv[closer], uu[closer]] = cc[closer]
    return img
