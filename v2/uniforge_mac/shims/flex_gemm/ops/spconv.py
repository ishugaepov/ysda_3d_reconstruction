"""Submanifold sparse 3D convolution (stride 1), same interface as flex_gemm.ops.spconv.

Weight layout (Co, Kw, Kh, Kd, Ci) and kernel-offset order match the CUDA kernels; neighbour maps
are cached by the caller exactly like FlexGEMM's.
"""
import torch


def set_algorithm(*_a, **_k):
    pass


def set_hashmap_ratio(*_a, **_k):
    pass


def _flat(coords, W, H, D):
    return ((coords[:, 0].long() * W + coords[:, 1].long()) * H + coords[:, 2].long()) * D + coords[:, 3].long()


def lookup(sorted_keys, order, query):
    """Index of each query key in the original array, or -1 (sorted-array hash map)."""
    pos = torch.searchsorted(sorted_keys, query).clamp_(max=len(sorted_keys) - 1)
    hit = sorted_keys[pos] == query
    return torch.where(hit, order[pos], torch.full_like(pos, -1))


def build_neighbor_map(coords, W, H, D, Kw, Kh, Kd, dilation):
    keys = _flat(coords, W, H, D)
    sorted_keys, order = torch.sort(keys)
    dx, dy, dz = dilation
    offs = []
    for vx in range(Kw):
        for vy in range(Kh):
            for vz in range(Kd):
                offs.append(((vx - Kw // 2) * dx, (vy - Kh // 2) * dy, (vz - Kd // 2) * dz))
    offs = torch.tensor(offs, device=coords.device, dtype=torch.long)       # (V,3)
    nb = coords[:, None, 1:].long() + offs[None]                            # (N,V,3)
    valid = ((nb >= 0) & (nb < torch.tensor([W, H, D], device=coords.device))).all(-1)
    q = ((coords[:, None, 0].long() * W + nb[..., 0]) * H + nb[..., 1]) * D + nb[..., 2]
    idx = lookup(sorted_keys, order, q.reshape(-1)).reshape(q.shape)
    return torch.where(valid, idx, torch.full_like(idx, -1))


def sparse_submanifold_conv3d(feats, coords, shape, weight, bias, neighbor_cache, dilation=(1, 1, 1),
                              chunk_rows: int = 65536):
    W, H, D = [int(s) for s in tuple(shape)[-3:]]
    Co, Kw, Kh, Kd, Ci = weight.shape
    V = Kw * Kh * Kd
    if feats.shape[0] == 0:
        return feats.new_empty((0, Co)), neighbor_cache
    nbr = neighbor_cache if neighbor_cache is not None else build_neighbor_map(
        coords, W, H, D, Kw, Kh, Kd, tuple(dilation))
    # fp32 matmul: MPS asserts on large-K bf16/fp16 matmuls ("Destination NDArray and Accumulator").
    w = weight.reshape(Co, V * Ci).t().float()
    out = torch.empty(feats.shape[0], Co, device=feats.device, dtype=feats.dtype)
    for a in range(0, feats.shape[0], chunk_rows):
        n = nbr[a:a + chunk_rows]
        g = feats[n.clamp(min=0)]
        g = g.masked_fill((n < 0)[..., None], 0).reshape(len(n), V * Ci).float()
        out[a:a + chunk_rows] = (g @ w).to(out.dtype)
    if bias is not None:
        out += bias.to(out.dtype)
    return out, nbr
