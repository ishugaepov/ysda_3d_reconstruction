"""Trilinear / nearest sampling of a sparse voxel field (flex_gemm.ops.grid_sample.grid_sample_3d).

Cell-centre convention: voxel c sits at continuous grid coordinate c + 0.5, so queries are shifted
by -0.5 before bracketing. Trilinear weights are renormalised over the occupied corners only.
"""
import torch

from .spconv import _flat, lookup


def grid_sample_3d(feats, coords, shape, grid, mode="trilinear"):
    """feats (N,C); coords (N,4) [b,x,y,z]; shape (..., W, H, D); grid (B,M,3) in voxel units -> (B,M,C)."""
    W, H, D = [int(s) for s in tuple(shape)[-3:]]
    sorted_keys, order = torch.sort(_flat(coords, W, H, D))
    B, M, _ = grid.shape
    out = []
    for b in range(B):
        g = grid[b].float() - 0.5
        if mode == "nearest":
            c = g.round().long()
            q = torch.cat([torch.full_like(c[:, :1], b), c], 1)
            idx = lookup(sorted_keys, order, _flat(q, W, H, D))
            val = feats[idx.clamp(min=0)] * (idx >= 0)[:, None]
            out.append(val)
            continue
        base = torch.floor(g).long()
        frac = g - base
        acc = torch.zeros(M, feats.shape[1], device=feats.device, dtype=torch.float32)
        wsum = torch.zeros(M, 1, device=feats.device, dtype=torch.float32)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    c = base + torch.tensor([dx, dy, dz], device=base.device)
                    w = ((frac[:, 0] if dx else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1])
                         * (frac[:, 2] if dz else 1 - frac[:, 2]))[:, None]
                    inb = ((c >= 0) & (c < torch.tensor([W, H, D], device=c.device))).all(-1)
                    q = torch.cat([torch.full_like(c[:, :1], b), c.clamp(min=0)], 1)
                    idx = lookup(sorted_keys, order, _flat(q, W, H, D))
                    ok = (inb & (idx >= 0))[:, None].float()
                    acc += feats[idx.clamp(min=0)].float() * w * ok
                    wsum += w * ok
        out.append((acc / wsum.clamp(min=1e-8)).to(feats.dtype))
    return torch.stack(out)
