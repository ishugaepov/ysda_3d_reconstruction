"""Per-stage diagnostics: which step of the pipeline loses quality?

All functions return images (float [0,1], H x W x 3) and plain-number stats, so they can be shown in
ComfyUI preview nodes and pasted into journal entries.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import recon
from .render import render_points


def _turbo(x: torch.Tensor) -> torch.Tensor:
    """Cheap turbo-like colormap for x in [0,1] → (...,3)."""
    x = x.clamp(0, 1)
    r = (1.5 - (4 * x - 3).abs()).clamp(0, 1)
    g = (1.5 - (4 * x - 2).abs()).clamp(0, 1)
    b = (1.5 - (4 * x - 1).abs()).clamp(0, 1)
    return torch.stack([r, g, b], -1)


def mask_overlay(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Object kept as is, background dimmed, mask boundary in yellow. (B,H,W,3)."""
    m = (masks >= 0.5).float()
    edge = (F.max_pool2d(m[:, None], 5, 1, 2) - (-F.max_pool2d(-m[:, None], 5, 1, 2)))[:, 0] > 0
    out = images * (0.25 + 0.75 * masks[..., None])
    out[edge] = torch.tensor([1.0, 0.9, 0.0])
    return out


def consistency_check(depth, K, E, images, keep, stride: int = 3, rel_tol: float = 0.02, max_points=600_000,
                      dist=None):
    """Project the masked points of all OTHER views into each view and compare with that view.

    With perfect poses and depth, other views' points land exactly on this view's object, at the
    same depth and with the same colours. Returns:
      tiles (S,H,3W,3): [photo | other views' points rendered here | depth disagreement heatmap]
      stats: per view, median |z_other - z_this| / z_this and the fraction above rel_tol.
    The heatmap shows blue = agree, red = >= 3*rel_tol; grey = only one source sees it.
    """
    S, H, W = depth.shape
    pts, cols, owner = [], [], []
    for j in range(S):
        X, (v, u) = recon.unproject(depth[j], K[j], E[j], keep[j], stride, recon._k1(dist, j))
        pts.append(X); cols.append(images[j][v, u]); owner.append(torch.full((len(X),), j))
    pts, cols, owner = torch.cat(pts), torch.cat(cols), torch.cat(owner)
    if len(pts) > max_points:
        sel = torch.randperm(len(pts))[:max_points]
        pts, cols, owner = pts[sel], cols[sel], owner[sel]
    tiles, stats = [], []
    for i in range(S):
        others = owner != i
        P = pts[others]
        Xc = P @ E[i][:3, :3].T + E[i][:3, 3]
        # Depth image of the other views' points (as colours = depth), via the splat renderer.
        k1 = recon._k1(dist, i)
        zimg = render_points(P, Xc[:, 2:3].repeat(1, 3), K[i], E[i], (H, W), radius=1, bg=np.inf, k1=k1)[..., 0]
        cimg = render_points(P, cols[others], K[i], E[i], (H, W), radius=1, bg=1.0, k1=k1)
        zimg = torch.from_numpy(zimg)
        both = keep[i] & torch.isfinite(zimg)
        rel = ((zimg - depth[i]).abs() / depth[i].clamp(min=1e-6))
        rel_both = rel[both]
        med = float(rel_both.median()) if both.any() else float("nan")
        frac = float((rel_both > rel_tol).float().mean()) if both.any() else float("nan")
        stats.append({"view": i, "median_rel_depth_err": round(med, 4), f"frac_over_{rel_tol}": round(frac, 3),
                      "overlap_px": int(both.sum())})
        heat = torch.full((H, W, 3), 0.85)
        heat[keep[i] | torch.isfinite(zimg)] = 0.55
        heat[both] = _turbo(rel[both] / (3 * rel_tol))
        tiles.append(torch.cat([images[i], torch.from_numpy(cimg), heat], 1))
    return torch.stack(tiles), stats, (pts, cols)


def silhouette_check(verts, faces, K, E, masks, n_samples=400_000, dist=None):
    """Compare the reconstructed shape's silhouette with each mask.

    Overlay: white = both, green = in mask but carved away (lost volume),
    magenta = mesh outside the mask (extra volume). Returns (S,H,W,3) and per-view IoU.
    """
    S, H, W = masks.shape
    tri = verts[faces]
    area = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    fi = torch.multinomial(area / area.sum(), n_samples, replacement=True)
    r1, r2 = torch.rand(n_samples).sqrt(), torch.rand(n_samples)
    w = torch.stack([1 - r1, r1 * (1 - r2), r1 * r2], -1)
    P = (tri[fi] * w[..., None]).sum(1)
    ones = torch.ones(len(P), 3)
    out, ious = [], []
    for i in range(S):
        sil = torch.from_numpy(render_points(P, ones, K[i], E[i], (H, W), radius=2, bg=0.0, k1=recon._k1(dist, i)))[..., 0] > 0.5
        sil = (-F.max_pool2d(-F.max_pool2d(sil.float()[None, None], 5, 1, 2), 5, 1, 2))[0, 0] > 0.5  # close holes
        m = masks[i] >= 0.5
        inter, union = (sil & m).sum(), (sil | m).sum()
        ious.append(round(float(inter) / max(1, float(union)), 3))
        img = torch.zeros(H, W, 3)
        img[sil & m] = 1.0
        img[m & ~sil] = torch.tensor([0.1, 0.9, 0.1])
        img[sil & ~m] = torch.tensor([0.9, 0.1, 0.9])
        out.append(img)
    return torch.stack(out), ious


def write_point_cloud_ply(path: str, points: torch.Tensor, colors: torch.Tensor, camera_centers=None):
    """Binary PLY with RGB. Cameras (if given) are added as small red point clusters."""
    P, C = points.numpy().astype(np.float32), (colors.numpy() * 255).clip(0, 255).astype(np.uint8)
    if camera_centers is not None:
        cc = camera_centers.numpy().astype(np.float32)
        s = float(np.ptp(P, 0).max()) * 0.01
        blob = (np.random.default_rng(0).normal(size=(60, 3)) * s).astype(np.float32)
        cams = (cc[:, None, :] + blob[None]).reshape(-1, 3)
        P = np.concatenate([P, cams]); C = np.concatenate([C, np.tile([[255, 0, 0]], (len(cams), 1)).astype(np.uint8)])
    # OpenCV (y down, z forward) → viewer convention (y up), same as recon.opencv_to_gltf.
    P = P * np.array([1, -1, -1], np.float32)
    rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = P.T
    rec["r"], rec["g"], rec["b"] = C.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode()); f.write(rec.tobytes())


def _fill_splat_gaps(img: torch.Tensor, cov: torch.Tensor, iters: int = 4):
    """Fill pixels inside the (closed) rendered silhouette that no splat hit, from covered neighbours.

    The point-splat renderer leaves sub-pixel gaps that show the white background; without this,
    meshes with more total area (e.g. generated meshes with inner shells) look brighter and score
    worse for a reason that has nothing to do with their colours (journal 0012).
    """
    closed = (-F.max_pool2d(-F.max_pool2d(cov.float()[None, None], 5, 1, 2), 5, 1, 2))[0, 0] > 0.5
    c = cov.float()
    x = img * c[..., None]
    for _ in range(iters):
        num = F.avg_pool2d(x.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
        den = F.avg_pool2d(c[None, None], 3, 1, 1)[0, 0]
        fill = closed & (c < 0.5) & (den > 0)
        x = torch.where(fill[..., None], num / den.clamp(min=1e-6)[..., None], x)
        c = torch.where(fill, torch.ones_like(c), c)
    return torch.where(c[..., None] > 0.5, x, img), c > 0.5


def photometric_check(verts, faces, colors, K, E, images, masks, views, dist=None, n_samples=None):
    """Render the vertex-coloured mesh into `views` and compare with the photos inside the mask.

    Returns per-view {psnr, coverage} (coverage = fraction of mask pixels the render covers) and
    tiles [photo | render | |error|×4] for inspection. Use held-out views for an honest number.
    """
    S, H, W = masks.shape
    if n_samples is None:
        # ~4 samples per pixel of the largest projected footprint, from the mesh area.
        tri = verts[faces]
        area = float(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1).sum() / 2)
        Cc = recon.camera_centers(E[list(views)])
        dmin = float((verts.mean(0) - Cc).norm(dim=-1).min())
        f = float(K[list(views)][:, 0, 0].max())
        n_samples = int(min(20_000_000, max(2_000_000, 4 * area * (f / dmin) ** 2)))
    P, C = recon.surface_samples(verts, faces, n_samples, colors)
    out, tiles = [], []
    for i in views:
        k1 = recon._k1(dist, i)
        zb = recon.zbuffer(P, K[i], E[i], (H, W), k1, fill=0)
        img = torch.from_numpy(render_points(P, C, K[i], E[i], (H, W), radius=1, bg=1.0, k1=k1))
        cov = torch.from_numpy(render_points(P, torch.ones_like(C), K[i], E[i], (H, W), radius=1, bg=0.0, k1=k1))[..., 0] > 0.5
        img, cov = _fill_splat_gaps(img, cov)
        m = (masks[i] >= 0.5) & cov
        mse = float(((img - images[i]) ** 2)[m].mean()) if m.any() else float("nan")
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        # Full-mask PSNR: every object pixel counts; uncovered ones are compared against the white
        # background, so holes and shrinkage are penalised. Comparable across methods.
        full = masks[i] >= 0.5
        mse_full = float(((img - images[i]) ** 2)[full].mean()) if full.any() else float("nan")
        out.append({"view": int(i), "psnr": round(psnr, 2), "psnr_full_mask": round(10 * np.log10(1.0 / max(mse_full, 1e-10)), 2),
                    "coverage": round(float(m.sum()) / max(1, int((masks[i] >= 0.5).sum())), 3)})
        err = ((img - images[i]).abs() * 4).clamp(0, 1) * m[..., None]
        tiles.append(torch.cat([images[i], img, err], 1))
    return out, torch.stack(tiles)
