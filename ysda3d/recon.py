"""v0 reconstruction: DA3 cameras + depth + object masks → coloured mesh.

Conventions (matching ComfyUI's Depth Anything 3 node):
- extrinsics E = [R|t] map world → camera, OpenCV camera axes (x right, y down, z forward);
- intrinsics K are in pixels of DA3's *processing* resolution and must be rescaled to the
  resolution of the depth maps (see `rescale_intrinsics`).

Surface method (v0): voxel carving. A voxel is kept if, in (almost) every view that sees it,
it projects inside the object mask and is not in front of the observed depth. The result is a
closed solid; the iso-surface is extracted by a caller-supplied function (ComfyUI's surface nets).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

PATCH = 14


def da3_processing_size(h: int, w: int, process_res: int = 504, method: str = "upper_bound_resize") -> tuple[int, int]:
    """Same rule as comfy/ldm/depth_anything_3/preprocess.compute_target_size."""
    def round_to_patch(x: int) -> int:
        down = (x // PATCH) * PATCH
        up = down + PATCH
        return up if abs(up - x) <= abs(x - down) else down
    s = process_res / float(max(h, w) if method == "upper_bound_resize" else min(h, w))
    return max(PATCH, round_to_patch(round(h * s))), max(PATCH, round_to_patch(round(w * s)))


def rescale_intrinsics(K: torch.Tensor, proc_hw: tuple[int, int], out_hw: tuple[int, int]) -> torch.Tensor:
    """(S,3,3) K at proc_hw → K at out_hw (pixel-centre offsets ignored at this precision)."""
    K = K.clone().float()
    K[:, 0, :] *= out_hw[1] / proc_hw[1]
    K[:, 1, :] *= out_hw[0] / proc_hw[0]
    return K


def camera_centers(E: torch.Tensor) -> torch.Tensor:
    R, t = E[:, :3, :3], E[:, :3, 3]
    return -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)


def project(X: torch.Tensor, K: torch.Tensor, E: torch.Tensor, k1: float = 0.0):
    """World points (N,3) → pixel coords (N,2) and camera depth z (N,) for one view.

    k1: one-parameter radial distortion (COLMAP SIMPLE_RADIAL): p_d = p (1 + k1 |p|²) on
    normalised coordinates p = (x/z, y/z).
    """
    Xc = X @ E[:3, :3].T + E[:3, 3]
    z = Xc[:, 2]
    p = Xc[:, :2] / z.clamp(min=1e-6)[:, None]
    if k1:
        p = p * (1 + k1 * (p * p).sum(-1, keepdim=True))
    u = K[0, 0] * p[:, 0] + K[0, 2]
    v = K[1, 1] * p[:, 1] + K[1, 2]
    return torch.stack([u, v], -1), z


def undistort_normalized(pd: torch.Tensor, k1: float, iters: int = 8) -> torch.Tensor:
    """Invert p_d = p (1 + k1 |p|²) by fixed-point iteration (fine for |k1| ~ 0.05)."""
    if not k1:
        return pd
    p = pd.clone()
    for _ in range(iters):
        p = pd / (1 + k1 * (p * p).sum(-1, keepdim=True))
    return p


def sample(img: torch.Tensor, uv: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample (H,W,C) at pixel coords (N,2) → (N,C); out-of-range samples return 0."""
    H, W = img.shape[:2]
    g = torch.stack([uv[:, 0] / (W - 1) * 2 - 1, uv[:, 1] / (H - 1) * 2 - 1], -1).view(1, 1, -1, 2)
    out = F.grid_sample(img.permute(2, 0, 1)[None].float(), g, mode=mode, align_corners=True)
    return out[0, :, 0].T


def unproject(depth: torch.Tensor, K: torch.Tensor, E: torch.Tensor, keep: torch.Tensor, stride: int = 4,
              k1: float = 0.0):
    """Masked depth of one view → world points (N,3) plus their pixel indices."""
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(0, H, stride), torch.arange(0, W, stride), indexing="ij")
    sel = keep[v, u] & (depth[v, u] > 0)
    u, v = u[sel].float(), v[sel].float()
    d = depth[v.long(), u.long()]
    p = undistort_normalized(torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]], -1), k1)
    Xc = torch.cat([p * d[:, None], d[:, None]], -1)
    R, t = E[:3, :3], E[:3, 3]
    return (Xc - t) @ R, (v.long(), u.long())


def _k1(dist, i):
    return float(dist[i]) if dist is not None else 0.0


def fused_points(depth, K, E, images, keep, stride: int = 4, dist=None):
    pts, cols = [], []
    for i in range(depth.shape[0]):
        X, (v, u) = unproject(depth[i], K[i], E[i], keep[i], stride, _k1(dist, i))
        pts.append(X)
        cols.append(images[i][v, u])
    return torch.cat(pts), torch.cat(cols)


def object_bounds(points: torch.Tensor, lo: float = 0.02, hi: float = 0.98, margin: float = 0.15):
    q = torch.quantile(points[torch.randperm(len(points))[:200_000]], torch.tensor([lo, hi]), dim=0)
    pad = (q[1] - q[0]) * margin
    return q[0] - pad, q[1] + pad


def carve(depth, K, E, masks, bmin, bmax, res: int = 192, depth_tol: float = 0.02,
          max_violations: int = 1, min_views: int = 3, chunk: int = 2_000_000, dist=None):
    """Voxel carving with mask and depth constraints.

    depth_tol is a fraction of the bounding-box diagonal: a voxel more than that far in front of
    the observed surface is free space. Returns occupancy (X,Y,Z) in {0,1}, voxel size, origin.
    """
    ext = bmax - bmin
    voxel = float(ext.max()) / res
    dims = [max(2, int(math.ceil(float(e) / voxel))) for e in ext]
    axes = [bmin[i] + (torch.arange(dims[i]) + 0.5) * voxel for i in range(3)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    tol = depth_tol * float(ext.norm())
    viol = torch.zeros(len(grid), dtype=torch.int16)
    seen = torch.zeros(len(grid), dtype=torch.int16)
    S, H, W = depth.shape
    for i in range(S):
        for a in range(0, len(grid), chunk):
            X = grid[a:a + chunk]
            uv, z = project(X, K[i], E[i], _k1(dist, i))
            inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
            m = sample(masks[i][..., None], uv, "nearest")[:, 0]
            d = sample(depth[i][..., None], uv, "nearest")[:, 0]
            free = (m < 0.5) | ((m >= 0.5) & (z < d - tol))
            seen[a:a + chunk] += inside.short()
            viol[a:a + chunk] += (inside & free).short()
    occ = (viol <= max_violations) & (seen >= min_views)
    return occ.view(*dims).float(), voxel, (bmin + 0.5 * voxel)


def smooth_volume(vol: torch.Tensor, iters: int = 1) -> torch.Tensor:
    v = vol[None, None]
    for _ in range(iters):
        v = F.avg_pool3d(v, 3, stride=1, padding=1)
    return v[0, 0]


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tri = verts[faces]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    n = torch.zeros_like(verts).index_add_(0, faces.reshape(-1), fn.repeat_interleave(3, 0))
    return F.normalize(n, dim=-1)


def largest_component(verts: np.ndarray, faces: np.ndarray):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    keep_label = np.bincount(lab[faces[:, 0]]).argmax()
    fmask = lab[faces[:, 0]] == keep_label
    faces = faces[fmask]
    used = np.unique(faces)
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def surface_samples(verts: torch.Tensor, faces: torch.Tensor, n: int, values: torch.Tensor | None = None,
                    seed: int = 0):
    """Area-weighted random points on the mesh, plus per-vertex `values` interpolated barycentrically."""
    g = torch.Generator().manual_seed(seed)
    tri = verts[faces]
    area = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    fi = torch.multinomial(area / area.sum(), n, replacement=True, generator=g)
    r1, r2 = torch.rand(n, generator=g).sqrt(), torch.rand(n, generator=g)
    w = torch.stack([1 - r1, r1 * (1 - r2), r1 * r2], -1)
    P = (tri[fi] * w[..., None]).sum(1)
    V = (values[faces[fi]] * w[..., None]).sum(1) if values is not None else None
    return P, V


def zbuffer(points: torch.Tensor, K: torch.Tensor, E: torch.Tensor, hw, k1: float = 0.0, fill: int = 1):
    """Nearest depth per pixel of a dense point set (a cheap mesh rasteriser); inf where empty.

    `fill` iterations of a 3x3 min-filter close the small gaps between splatted samples.
    """
    H, W = hw
    uv, z = project(points, K, E, k1)
    u, v = uv[:, 0].round().long(), uv[:, 1].round().long()
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zb = torch.full((H * W,), float("inf"))
    zb.scatter_reduce_(0, (v[ok] * W + u[ok]), z[ok], reduce="amin")
    zb = zb.view(H, W)
    for _ in range(fill):
        big = torch.where(torch.isfinite(zb), zb, torch.full_like(zb, 1e9))
        mn = -F.max_pool2d(-big[None, None], 3, 1, 1)[0, 0]
        zb = torch.where(torch.isfinite(zb), zb, torch.where(mn < 1e9, mn, zb))
    return zb


def mesh_depth_maps(verts, faces, K, E, hw, dist=None, n_samples: int = 2_000_000):
    """Per-view depth of the reconstructed mesh itself (S,H,W), for visibility tests."""
    P, _ = surface_samples(verts, faces, n_samples)
    return torch.stack([zbuffer(P, K[i], E[i], hw, _k1(dist, i)) for i in range(len(K))])


def erode_masks(masks: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return masks
    m = (masks >= 0.5).float()[:, None]
    return (-F.max_pool2d(-m, 2 * px + 1, 1, px))[:, 0]


def vertex_colors(verts, normals, depth, K, E, images, masks, vis_tol: float, power: float = 2.0, dist=None):
    """Colour each vertex from the views that see it, weighted by how frontally they see it.

    Visible = projects inside the mask and lies within vis_tol of `depth` at that pixel. `depth` is
    either DA3's depth (v0/v1) or the mesh's own z-buffer (v1.1, `mesh_depth_maps`), which makes the
    test consistent with the surface being coloured. Higher `power` → closer to "best view only".
    Vertices no view sees get the colour of the nearest coloured vertex.
    """
    centers = camera_centers(E)
    acc = torch.zeros(len(verts), 3)
    wsum = torch.zeros(len(verts))
    S, H, W = depth.shape
    for i in range(S):
        uv, z = project(verts, K[i], E[i], _k1(dist, i))
        inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
        d = sample(depth[i][..., None], uv, "nearest")[:, 0]
        m = sample(masks[i][..., None], uv, "nearest")[:, 0]
        vis = inside & (m >= 0.5) & ((z - d).abs() < vis_tol)
        to_cam = F.normalize(centers[i] - verts, dim=-1)
        w = (normals * to_cam).sum(-1).clamp(min=0) ** power * vis.float()
        acc += sample(images[i], uv) * w[:, None]
        wsum += w
    colors = acc / wsum.clamp(min=1e-8)[:, None]
    seen = wsum > 1e-6
    if (~seen).any() and seen.any():
        from scipy.spatial import cKDTree
        _, idx = cKDTree(verts[seen].numpy()).query(verts[~seen].numpy())
        colors[~seen] = colors[seen][torch.as_tensor(idx)]
    return colors.clamp(0, 1), float(seen.float().mean())


def opencv_to_gltf(v: torch.Tensor) -> torch.Tensor:
    """OpenCV world (y down, z forward) → glTF (y up, z towards viewer)."""
    return v * torch.tensor([1.0, -1.0, -1.0])


def reconstruct(depth, K, E, images, masks, surface_fn, res: int = 192, depth_tol: float = 0.02,
                max_violations: int = 1, conf=None, conf_quantile: float = 0.1, smooth_iters: int = 1,
                dist=None, visibility: str = "da3", color_power: float = 2.0, mask_erode_px: int = 0,
                log=print):
    """Full v0: returns (verts in DA3 world space (V,3), faces (F,3), colors (V,3), report dict)."""
    keep = masks >= 0.5
    if conf is not None:
        thr = torch.quantile(conf[keep][torch.randperm(int(keep.sum()))[:1_000_000]], conf_quantile)
        keep = keep & (conf >= thr)
    pts, _ = fused_points(depth, K, E, images, keep, dist=dist)
    if len(pts) < 100:
        raise ValueError("too few object points: check the masks (is the object segmented?)")
    bmin, bmax = object_bounds(pts)
    log(f"fused {len(pts)} points; bounds {bmin.tolist()} .. {bmax.tolist()}")
    occ, voxel, origin = carve(depth, K, E, masks, bmin, bmax, res, depth_tol, max_violations, dist=dist)
    log(f"carved grid {tuple(occ.shape)}: {int(occ.sum())} occupied voxels")
    if occ.sum() == 0:
        raise ValueError("carving removed everything: masks or camera poses are inconsistent")
    vol = smooth_volume(occ, smooth_iters)
    verts, faces = surface_fn(vol, 0.5, voxel, origin.tolist())
    verts, faces = np.asarray(verts, np.float32), np.asarray(faces, np.int64)
    if len(faces) == 0:
        raise ValueError("no surface extracted")
    verts, faces = largest_component(verts, faces)
    verts_t, faces_t = torch.from_numpy(verts), torch.from_numpy(faces)
    normals = vertex_normals(verts_t, faces_t)
    # Make normals point outward (away from the centroid) on average.
    if ((verts_t - verts_t.mean(0)) * normals).sum() < 0:
        faces_t = faces_t[:, [0, 2, 1]]
        normals = -normals
    S, H, W = depth.shape
    if visibility == "mesh":
        vis_depth, vis_tol = mesh_depth_maps(verts_t, faces_t, K, E, (H, W), dist), 1.5 * voxel
    else:
        vis_depth, vis_tol = depth, 4 * voxel
    colors, seen_frac = vertex_colors(verts_t, normals, vis_depth, K, E, images,
                                      erode_masks(masks, mask_erode_px), vis_tol=vis_tol,
                                      power=color_power, dist=dist)
    report = {
        "views": int(depth.shape[0]), "fused_points": int(len(pts)), "grid": list(occ.shape),
        "voxel_size": voxel, "vertices": int(len(verts_t)), "faces": int(len(faces_t)),
        "vertices_seen_fraction": round(seen_frac, 3),
    }
    return verts_t, faces_t, colors, report


def to_display(verts: torch.Tensor) -> torch.Tensor:
    """World (OpenCV) vertices → centred, unit-size, glTF-oriented vertices for viewers.

    (x,-y,-z) is a 180° rotation about x (det +1), so face winding stays outward.
    """
    v = opencv_to_gltf(verts - verts.mean(0))
    return v / v.abs().max()
