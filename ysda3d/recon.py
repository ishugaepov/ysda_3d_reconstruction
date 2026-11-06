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


def project(X: torch.Tensor, K: torch.Tensor, E: torch.Tensor):
    """World points (N,3) → pixel coords (N,2) and camera depth z (N,) for one view."""
    Xc = X @ E[:3, :3].T + E[:3, 3]
    z = Xc[:, 2]
    zs = z.clamp(min=1e-6)
    u = K[0, 0] * Xc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zs + K[1, 2]
    return torch.stack([u, v], -1), z


def sample(img: torch.Tensor, uv: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample (H,W,C) at pixel coords (N,2) → (N,C); out-of-range samples return 0."""
    H, W = img.shape[:2]
    g = torch.stack([uv[:, 0] / (W - 1) * 2 - 1, uv[:, 1] / (H - 1) * 2 - 1], -1).view(1, 1, -1, 2)
    out = F.grid_sample(img.permute(2, 0, 1)[None].float(), g, mode=mode, align_corners=True)
    return out[0, :, 0].T


def unproject(depth: torch.Tensor, K: torch.Tensor, E: torch.Tensor, keep: torch.Tensor, stride: int = 4):
    """Masked depth of one view → world points (N,3) plus their pixel indices."""
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(0, H, stride), torch.arange(0, W, stride), indexing="ij")
    sel = keep[v, u] & (depth[v, u] > 0)
    u, v = u[sel].float(), v[sel].float()
    d = depth[v.long(), u.long()]
    Xc = torch.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
    R, t = E[:3, :3], E[:3, 3]
    return (Xc - t) @ R, (v.long(), u.long())


def fused_points(depth, K, E, images, keep, stride: int = 4):
    pts, cols = [], []
    for i in range(depth.shape[0]):
        X, (v, u) = unproject(depth[i], K[i], E[i], keep[i], stride)
        pts.append(X)
        cols.append(images[i][v, u])
    return torch.cat(pts), torch.cat(cols)


def object_bounds(points: torch.Tensor, lo: float = 0.02, hi: float = 0.98, margin: float = 0.15):
    q = torch.quantile(points[torch.randperm(len(points))[:200_000]], torch.tensor([lo, hi]), dim=0)
    pad = (q[1] - q[0]) * margin
    return q[0] - pad, q[1] + pad


def carve(depth, K, E, masks, bmin, bmax, res: int = 192, depth_tol: float = 0.02,
          max_violations: int = 1, min_views: int = 3, chunk: int = 2_000_000):
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
            uv, z = project(X, K[i], E[i])
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


def vertex_colors(verts, normals, depth, K, E, images, masks, vis_tol: float, power: float = 2.0):
    """Colour each vertex from the views that see it, weighted by how frontally they see it.

    Visible = projects inside the mask and lies within vis_tol of that view's observed depth.
    Vertices no view sees get the colour of the nearest coloured vertex.
    """
    centers = camera_centers(E)
    acc = torch.zeros(len(verts), 3)
    wsum = torch.zeros(len(verts))
    S, H, W = depth.shape
    for i in range(S):
        uv, z = project(verts, K[i], E[i])
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
                log=print):
    """Full v0: returns (verts in DA3 world space (V,3), faces (F,3), colors (V,3), report dict)."""
    keep = masks >= 0.5
    if conf is not None:
        thr = torch.quantile(conf[keep][torch.randperm(int(keep.sum()))[:1_000_000]], conf_quantile)
        keep = keep & (conf >= thr)
    pts, _ = fused_points(depth, K, E, images, keep)
    if len(pts) < 100:
        raise ValueError("too few object points: check the masks (is the object segmented?)")
    bmin, bmax = object_bounds(pts)
    log(f"fused {len(pts)} points; bounds {bmin.tolist()} .. {bmax.tolist()}")
    occ, voxel, origin = carve(depth, K, E, masks, bmin, bmax, res, depth_tol, max_violations)
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
    colors, seen_frac = vertex_colors(verts_t, normals, depth, K, E, images, masks,
                                      vis_tol=4 * voxel)
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
