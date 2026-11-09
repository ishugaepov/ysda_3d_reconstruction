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


def edge_weight(masks: torch.Tensor, edge_px: int) -> torch.Tensor:
    """~0 at the mask boundary rising to 1 at edge_px inside (multi-radius erosion average)."""
    if edge_px <= 0:
        return (masks >= 0.5).float()
    m = (masks >= 0.5).float()[:, None]
    radii = sorted({max(1, round(edge_px * k / 4)) for k in range(1, 5)})
    acc = torch.zeros_like(m)
    for r in radii:
        acc += -F.max_pool2d(-m, 2 * r + 1, 1, r)
    return (acc / len(radii))[:, 0]


def tsdf_in_hull(depth, K, E, masks, hull, voxel: float, origin, conf=None, trunc_voxels: float = 5.0,
                 chunk: int = 2_000_000, dist=None, edge_px: int = 0, bias_voxels: float = 0.0):
    """Truncated signed-distance fusion of the (rescaled) depth maps, restricted to a hull.

    For each voxel x and view i where x projects inside the mask: sdf_i = d_i(u,v) - z_i(x)
    (positive = in front of the observed surface). Views with sdf_i > -trunc contribute
    clamp(sdf_i/trunc, -1, 1), weighted by the normalised DA3 confidence at (u,v) (Curless & Levoy
    1996). Voxels inside the hull that no view observes near the surface are interior (occluded,
    e.g. the underside); voxels outside the hull are empty. Returns an occupancy-like field in [0,1]
    (0.5 = surface) for the iso-surface extractor.
    """
    dims = hull.shape
    axes = [origin[i] + torch.arange(dims[i]) * voxel for i in range(3)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside_hull = hull.reshape(-1) > 0.5
    trunc = trunc_voxels * voxel
    num = torch.zeros(len(grid))
    den = torch.zeros(len(grid))
    S, H, W = depth.shape
    cnorm = None
    if conf is not None:
        cnorm = torch.stack([(c / torch.quantile(c[masks[i] >= 0.5], 0.9).clamp(min=1e-6)).clamp(0, 1)
                             for i, c in enumerate(conf)])
    ew = edge_weight(masks, edge_px) if edge_px > 0 else None
    idx = torch.nonzero(inside_hull)[:, 0]
    for i in range(S):
        for a in range(0, len(idx), chunk):
            ii = idx[a:a + chunk]
            uv, z = project(grid[ii], K[i], E[i], _k1(dist, i))
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
            m = sample(masks[i][..., None], uv, "nearest")[:, 0] >= 0.5
            d = sample(depth[i][..., None], uv, "nearest")[:, 0]
            sdf = d - z
            use = ok & m & (sdf > -trunc)
            w = sample(cnorm[i][..., None], uv, "nearest")[:, 0] if cnorm is not None else torch.ones_like(z)
            if ew is not None:
                w = w * sample(ew[i][..., None], uv, "nearest")[:, 0]
            w = w * use.float()
            num[ii] += w * ((sdf - bias_voxels * voxel) / trunc).clamp(-1, 1)
            den[ii] += w
    tsdf = torch.where(den > 1e-6, num / den.clamp(min=1e-6), torch.full_like(num, -1.0))  # unobserved interior
    tsdf = torch.where(inside_hull, tsdf, torch.ones_like(tsdf))                           # outside hull: empty
    return ((1 - tsdf) / 2).view(*dims)


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
                surface_mode: str = "carve", tsdf_trunc_voxels: float = 5.0, tsdf_edge_px: int = 0,
                tsdf_bias_voxels: float = 0.0, tsdf_calibrate: bool = False, log=print):
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
    calib = None
    if tsdf_calibrate:
        # Per-view depth scale against the visual hull's own depth (exact where the object is
        # convex). Removes a systematic ~4% DA3 depth bias on the toy (journal 0008).
        hull, voxel, origin = carve(depth, K, E, masks, bmin, bmax, res, 10.0, 1, dist=dist)
        hv, hf = surface_fn(smooth_volume(hull, smooth_iters), 0.5, voxel, origin.tolist())
        hv, hf = largest_component(np.asarray(hv, np.float32), np.asarray(hf, np.int64))
        hz = mesh_depth_maps(torch.from_numpy(hv), torch.from_numpy(hf), K, E, depth.shape[1:], dist)
        inner = erode_masks(masks, 5) >= 0.5
        calib = []
        for i in range(depth.shape[0]):
            ok = inner[i] & torch.isfinite(hz[i]) & (depth[i] > 0)
            calib.append(float((hz[i][ok] / depth[i][ok]).median()) if ok.sum() > 100 else 1.0)
        depth = depth * torch.tensor(calib)[:, None, None]
        log(f"hull depth calibration factors: {[round(c, 4) for c in calib]}")
    carve_tol = 10.0 if surface_mode == "tsdf_in_hull" else depth_tol
    occ, voxel, origin = carve(depth, K, E, masks, bmin, bmax, res, carve_tol, max_violations, dist=dist)
    if surface_mode == "tsdf_in_hull":
        # occ is the visual hull; the surface inside it comes from depth fusion.
        field = tsdf_in_hull(depth, K, E, masks, occ, voxel, origin, conf=conf,
                             trunc_voxels=tsdf_trunc_voxels, dist=dist, edge_px=tsdf_edge_px,
                             bias_voxels=tsdf_bias_voxels)
        log(f"TSDF in hull: {int((field >= 0.5).sum())} interior voxels")
    log(f"carved grid {tuple(occ.shape)}: {int(occ.sum())} occupied voxels")
    if occ.sum() == 0:
        raise ValueError("carving removed everything: masks or camera poses are inconsistent")
    vol = smooth_volume(field if surface_mode == "tsdf_in_hull" else occ, smooth_iters)
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
    if calib is not None:
        report["hull_depth_calibration"] = [round(c, 4) for c in calib]
    return verts_t, faces_t, colors, report


def floor_plane(depth, K, E, masks, dist=None, ring_inner_px: int = 15, ring_outer_px: int = 80,
                stride: int = 4, iters: int = 400, inlier_frac_tol: float = 0.01, seed: int = 0):
    """RANSAC plane through depth points in a ring around each mask (the surface the object stands on).

    Returns (normal n pointing towards the cameras, offset c with n·x = c, inlier fraction), or None.
    """
    m = (masks >= 0.5).float()[:, None]
    outer = F.max_pool2d(m, 2 * ring_outer_px + 1, 1, ring_outer_px)[:, 0] > 0
    inner = F.max_pool2d(m, 2 * ring_inner_px + 1, 1, ring_inner_px)[:, 0] > 0
    ring = outer & ~inner
    pts = []
    for i in range(depth.shape[0]):
        X, _ = unproject(depth[i], K[i], E[i], ring[i], stride, _k1(dist, i))
        pts.append(X)
    P = torch.cat(pts)
    if len(P) < 100:
        return None
    g = torch.Generator().manual_seed(seed)
    P = P[torch.randperm(len(P), generator=g)[:200_000]]
    scale = float((P - P.median(0).values).norm(dim=-1).median())
    tol = inlier_frac_tol * scale * 2
    best = (0, None)
    for _ in range(iters):
        a, b, c = P[torch.randint(len(P), (3,), generator=g)]
        n = torch.cross(b - a, c - a, dim=-1)
        if n.norm() < 1e-9:
            continue
        n = n / n.norm()
        cnt = int(((P - a) @ n).abs().lt(tol).sum())
        if cnt > best[0]:
            best = (cnt, (n, float(a @ n)))
    if best[1] is None:
        return None
    n, off = best[1]
    inl = ((P @ n - off).abs() < tol)
    # Least-squares refit on inliers.
    Q = P[inl]
    mu = Q.mean(0)
    n = torch.linalg.svd(Q - mu, full_matrices=False).Vh[-1]
    off = float(mu @ n)
    if float((camera_centers(E) @ n - off).mean()) < 0:   # point the normal towards the cameras (up)
        n, off = -n, -off
    return n, off, float(inl.float().mean())


def upright_frame(E: torch.Tensor, up: torch.Tensor | None, anchor: int = 0):
    """Rotation R (rows = right, up, front) taking world vectors into an upright display frame.

    up: floor normal (or None → mean camera up-vector, i.e. -y of each camera). front: direction from
    the scene towards the anchor camera, projected onto the floor plane.
    """
    if up is None:
        up = F.normalize((-E[:, 1, :3]).mean(0), dim=0)       # camera -y axis in world coordinates
    up = F.normalize(up, dim=0)
    C = camera_centers(E)
    f = C[anchor] - C.mean(0)
    f = f - (f @ up) * up
    if f.norm() < 1e-9:
        f = F.normalize(torch.cross(up, torch.tensor([1.0, 0.0, 0.0]), dim=0), dim=0)
    front = F.normalize(f, dim=0)
    right = torch.cross(up, front, dim=0)                       # right-handed: x = y × z
    return torch.stack([right, up, front])


def to_display_upright(verts: torch.Tensor, R: torch.Tensor, floor_offset: float | None = None,
                       up: torch.Tensor | None = None) -> torch.Tensor:
    """Rotate into the upright frame (glTF: +Y up, +Z towards the anchor camera), unit size, standing
    on y = 0 (on the floor plane if given, else on the lowest vertex), centred in x/z."""
    v = verts @ R.T
    v = v - torch.stack([v[:, 0].mean(), torch.tensor(0.0), v[:, 2].mean()])
    base = (floor_offset if (floor_offset is not None) else None)
    y0 = float(v[:, 1].min()) if base is None else float(base)
    v[:, 1] -= y0
    return v / v.abs().max()


def to_display(verts: torch.Tensor) -> torch.Tensor:
    """World (OpenCV) vertices → centred, unit-size, glTF-oriented vertices for viewers.

    (x,-y,-z) is a 180° rotation about x (det +1), so face winding stays outward.
    """
    v = opencv_to_gltf(verts - verts.mean(0))
    return v / v.abs().max()
