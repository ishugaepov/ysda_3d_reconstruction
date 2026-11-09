"""Watertight remeshing of messy generated meshes (replaces CuMesh's narrow-band remesh on Mac).

UniForge3D's dual-grid output (via our Mac port, which skips CuMesh remeshing) is non-manifold:
thousands of tiny components, many edges shared by >2 faces, small holes (journal 0012). This turns it
into a closed manifold surface: voxelise the surface, dilate to bridge small gaps, flood-fill the
outside, erode back, extract an iso-surface. Colours are carried over by nearest original vertex.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree

from . import recon


def remesh_watertight(verts: torch.Tensor, faces: torch.Tensor, surface_fn, colors: torch.Tensor | None = None,
                      res: int = 256, close_iters: int = 2, solid_close_iters: int = 3,
                      samples_per_voxel: float = 8.0, log=print):
    v = verts.float()
    lo, hi = v.min(0).values, v.max(0).values
    pad = 3 + close_iters
    voxel = float((hi - lo).max()) / (res - 2 * pad)
    origin = lo - pad * voxel
    dims = [int(np.ceil(float(e) / voxel)) + 2 * pad for e in (hi - lo)]
    # Surface voxelisation by dense sampling (enough samples that every surface voxel is hit).
    area = float(torch.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]], dim=-1).norm(dim=-1).sum() / 2)
    n = int(min(30_000_000, max(1_000_000, samples_per_voxel * area / voxel ** 2)))
    P, _ = recon.surface_samples(v, faces.long(), n)
    ijk = ((P - origin) / voxel).floor().long()
    ijk = torch.minimum(ijk.clamp(min=0), torch.tensor(dims) - 1).numpy()
    surf = np.zeros(dims, bool)
    surf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    struct = ndimage.generate_binary_structure(3, 1)
    shell = ndimage.binary_dilation(surf, struct, iterations=close_iters)
    lab, _ = ndimage.label(~shell, struct)
    outside_labels = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                               lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    outside = np.isin(lab, outside_labels[outside_labels > 0])
    solid = ~outside
    if close_iters > 0:
        solid = ndimage.binary_erosion(solid, struct, iterations=close_iters, border_value=0)
    if solid_close_iters > 0:
        # Seal thin slots that connect to the outside (e.g. between the toy's face plate and body):
        # flood fill treats them as outside, and both slot walls would become surface (journal 0012).
        solid = ndimage.binary_closing(np.pad(solid, solid_close_iters), struct, iterations=solid_close_iters)[
            solid_close_iters:-solid_close_iters, solid_close_iters:-solid_close_iters, solid_close_iters:-solid_close_iters]
    log(f"   remesh: grid {dims}, {n/1e6:.1f}M samples, solid voxels {int(solid.sum())}")
    vol = recon.smooth_volume(torch.from_numpy(solid.astype(np.float32)), 1)
    nv, nf = surface_fn(vol, 0.5, voxel, (origin + 0.5 * voxel).tolist())
    nv, nf = recon.largest_component(np.asarray(nv, np.float32), np.asarray(nf, np.int64))
    nv_t, nf_t = torch.from_numpy(nv), torch.from_numpy(nf)
    new_colors = None
    if colors is not None:
        new_colors = transfer_colors(v, faces, colors, nv_t, nf_t, max_dist=3 * voxel)
    return nv_t, nf_t, new_colors


def transfer_colors(src_v, src_f, src_c, dst_v, dst_f, k: int = 16, max_dist: float | None = None):
    """Per-vertex colour transfer that respects orientation.

    Plain nearest-vertex transfer picks up colours from inner shells of a messy mesh (e.g. face
    colours on the back of the toy, journal 0012). Among the k nearest source vertices, prefer those
    whose normal agrees with the destination's outward normal; fall back to the nearest one.
    """
    sn = recon.vertex_normals(src_v.float(), src_f.long())
    dn = recon.vertex_normals(dst_v.float(), dst_f.long())
    if ((dst_v - dst_v.mean(0)) * dn).sum() < 0:
        dn = -dn
    dist, idx = cKDTree(src_v.numpy()).query(dst_v.numpy(), k=k)
    idx_t = torch.from_numpy(idx)
    agree = (sn[idx_t] * dn[:, None, :]).sum(-1)                     # (N,k) cos between normals
    d = torch.from_numpy(dist).float()
    score = torch.where(agree > 0.3, -d, torch.full_like(d, -1e9))  # nearest among agreeing candidates
    if max_dist is not None:
        score = torch.where(d <= max_dist, score, torch.full_like(score, -1e9))
    best = score.argmax(1)
    ok = score.max(1).values > -1e8
    chosen = torch.where(ok, idx_t[torch.arange(len(idx_t)), best], idx_t[:, 0])
    return src_c[chosen]


def cull_hidden_faces(verts: torch.Tensor, faces: torch.Tensor, n_dirs: int = 96, res: int = 768,
                      samples_per_face: int = 3, tol_frac: float = 0.002, extra_cams=None, log=print):
    """Keep only faces that are frontmost from at least one viewpoint (removes inner shells/buried fins).

    Viewpoints: a Fibonacci sphere of n_dirs orthographic directions around the mesh, plus optional real
    cameras (list of (K, E, (H, W))). A face counts as visible if any of its sample points matches the
    z-buffer at its pixel within tol_frac of the mesh size.
    """
    v, f = verts.float(), faces.long()
    c = (v.max(0).values + v.min(0).values) / 2
    ext = float((v.max(0).values - v.min(0).values).norm())
    g = torch.Generator().manual_seed(0)
    w = torch.rand(len(f), samples_per_face, 3, generator=g)
    w = w / w.sum(-1, keepdim=True)
    P = (v[f][:, None, :, :] * w[..., None]).sum(2).reshape(-1, 3)        # (F*s, 3)
    P = torch.cat([P, v[f].mean(1)])                                       # + centroids
    fid = torch.cat([torch.arange(len(f)).repeat_interleave(samples_per_face), torch.arange(len(f))])
    visible = torch.zeros(len(f), dtype=torch.bool)
    tol = tol_frac * ext
    i = torch.arange(n_dirs, dtype=torch.float32) + 0.5
    phi = torch.arccos(1 - 2 * i / n_dirs)
    theta = math.pi * (1 + 5 ** 0.5) * i
    dirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], -1)

    def mark(uv, z, H, W):
        u, vv = uv[:, 0].round().long(), uv[:, 1].round().long()
        ok = (z > 0) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        pix = vv[ok] * W + u[ok]
        zb = torch.full((H * W,), float("inf"))
        zb.scatter_reduce_(0, pix, z[ok], reduce="amin")
        hit = (z[ok] - zb[pix]) < tol
        visible[fid[ok][hit]] = True

    for d in dirs:                                   # orthographic views
        a = torch.tensor([0.0, 1.0, 0.0]) if abs(float(d[1])) < 0.9 else torch.tensor([1.0, 0.0, 0.0])
        x = F.normalize(torch.cross(a, d, dim=0), dim=0)
        y = torch.cross(d, x, dim=0)
        q = P - c
        uv = torch.stack([(q @ x) / ext * res + res / 2, (q @ y) / ext * res + res / 2], -1)
        mark(uv, q @ d + ext, res, res)
    for K, E, hw in (extra_cams or []):              # real cameras (pinhole)
        uv, z = recon.project(P, K, E)
        mark(uv, z, *hw)
    log(f"   cull: kept {int(visible.sum())} / {len(f)} faces")
    used = torch.unique(f[visible])
    remap = torch.full((len(v),), -1, dtype=torch.long)
    remap[used] = torch.arange(len(used))
    return v[used], remap[f[visible]], used
