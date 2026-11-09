"""v2.2/v2.3: bring a generated mesh (UniForge3D) into the COLMAP camera frame, then colour it from photos.

register_generated_mesh: 7-DoF similarity (scale, rotation, translation) that makes the generated mesh's
silhouettes match the object masks in all registered photos. Initialised from the upright frame
(floor plane + anchor camera), the same convention UniForge3D's output uses (+Y up, +Z = anchor/front).

recolour_from_photos: per-vertex colours from the photos that see each vertex (mesh z-buffer
visibility, cos² weighting, as v1.1); the generator's own colours are kept where no photo sees the
surface, with a soft blend at the boundary.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import recon


def _axis_angle_matrix(w: torch.Tensor) -> torch.Tensor:
    th = w.norm() + 1e-12
    k = w / th
    Kx = torch.zeros(3, 3, dtype=w.dtype)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, dtype=w.dtype) + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def _project_uv(X, K, E, k1):
    Xc = X @ E[:3, :3].T + E[:3, 3]
    z = Xc[:, 2].clamp(min=1e-6)
    p = Xc[:, :2] / z[:, None]
    if k1:
        p = p * (1 + k1 * (p * p).sum(-1, keepdim=True))
    return torch.stack([K[0, 0] * p[:, 0] + K[0, 2], K[1, 1] * p[:, 1] + K[1, 2]], -1)


def _outside_distance(masks: torch.Tensor) -> torch.Tensor:
    """Per-view distance (pixels) from each pixel to the mask; 0 inside."""
    from scipy.ndimage import distance_transform_edt
    return torch.stack([torch.from_numpy(distance_transform_edt(~(m.numpy() >= 0.5)).astype(np.float32))
                        for m in masks])


def silhouette_iou(verts, faces, K, E, masks, dist=None):
    from .diagnostics import silhouette_check
    _, ious = silhouette_check(verts, faces, K, E, masks, dist=dist)
    return ious


def register_generated_mesh(gen_verts, gen_faces, K, E, masks, dist=None, ref_verts=None,
                            floor=None, anchor: int = 0, iters: int = 300, n_points: int = 6000,
                            n_mask: int = 1500, yaw_search: bool = True, log=print):
    """Similarity transform taking generator-frame vertices (glTF: +Y up, +Z front) into the camera world.

    ref_verts: a reconstruction in the camera world (e.g. v1.2), used for the initial centre and scale.
    Returns (world_verts, info dict with scale/R/t and IoU before/after).
    """
    # ComfyUI runs nodes under torch.inference_mode(). Tensors created in it (the inputs, and anything
    # derived from them there) can never be saved for backward, and enable_grad() doesn't lift that.
    # So fit outside inference mode, on normal copies of the inputs (journal 0012).
    with torch.inference_mode(False):
        c = lambda x: x.clone() if torch.is_tensor(x) else x  # noqa: E731
        floor = tuple(c(x) for x in floor) if floor is not None else None
        return _register(c(gen_verts), c(gen_faces), c(K), c(E), c(masks), c(dist), c(ref_verts), floor,
                         anchor, iters, n_points, n_mask, yaw_search, log)


def _register(gen_verts, gen_faces, K, E, masks, dist, ref_verts, floor, anchor, iters, n_points, n_mask,
              yaw_search, log):
    S, H, W = masks.shape
    up = floor[0] if floor is not None else None
    R_up = recon.upright_frame(E, up, anchor)              # rows: right, up, front (world)
    g = gen_verts.float()
    g_c = (g.max(0).values + g.min(0).values) / 2
    g_h = float((g.max(0).values - g.min(0).values)[1])    # generator height along +Y
    if ref_verts is not None:
        r_up = ref_verts @ R_up.T
        r_c = (r_up.max(0).values + r_up.min(0).values) / 2
        s0 = float((r_up.max(0).values - r_up.min(0).values)[1]) / max(g_h, 1e-9)
        c_world = r_c @ R_up                               # back to world
    else:
        raise ValueError("ref_verts needed for initial scale (pass the v1 reconstruction)")

    # Fixed surface samples (barycentric), transformed by the current similarity each step.
    P0, _ = recon.surface_samples(g, gen_faces.long(), n_points)
    P0 = P0 - g_c
    D = _outside_distance(masks)
    diag = math.hypot(H, W)
    rng = torch.Generator().manual_seed(0)
    mask_px = []
    for i in range(S):
        ys, xs = torch.nonzero(masks[i] >= 0.5, as_tuple=True)
        sel = torch.randint(len(ys), (min(n_mask, len(ys)),), generator=rng)
        mask_px.append(torch.stack([xs[sel], ys[sel]], -1).float())

    def world(params, P):
        log_s, w, t = params
        Rg = _axis_angle_matrix(w)
        # generator frame -> upright world frame (R_up^T) -> camera world
        return (s0 * torch.exp(log_s)) * (P @ Rg.T) @ R_up + c_world + t

    def loss_fn(params):
        Xw = world(params, P0)
        total = 0.0
        for i in range(S):
            uv = _project_uv(Xw, K[i], E[i], recon._k1(dist, i))
            gx = uv[:, 0] / (W - 1) * 2 - 1
            gy = uv[:, 1] / (H - 1) * 2 - 1
            out = F.grid_sample(D[i][None, None], torch.stack([gx, gy], -1)[None, None], align_corners=True)[0, 0, 0]
            l_out = out.mean() / diag                                         # points outside the mask
            sub = uv[torch.randperm(len(uv), generator=rng)[:2000]]
            l_cov = torch.cdist(mask_px[i], sub).min(1).values.mean() / diag  # mask not covered
            total = total + l_out + l_cov
        return total / S

    def make(log_s=0.0, yaw=0.0):
        return [torch.tensor(log_s, requires_grad=True), torch.tensor([0.0, yaw, 1e-6], requires_grad=True),
                torch.zeros(3, requires_grad=True)]

    # Coarse yaw search (the generator's front is the anchor's azimuth, but be robust to a wrong guess).
    best = (float("inf"), 0.0)
    if yaw_search:
        with torch.no_grad():
            for yaw in np.linspace(-math.pi, math.pi, 12, endpoint=False):
                L = float(loss_fn(make(0.0, float(yaw))))
                if L < best[0]:
                    best = (L, float(yaw))
    params = make(0.0, best[1])
    with torch.no_grad():
        init_world = world(params, g - g_c)
    iou0 = silhouette_iou(init_world, gen_faces.long(), K, E, masks, dist)
    opt = torch.optim.Adam(params, lr=1e-2)
    with torch.enable_grad():                                 # callers may have disabled autograd
        for it in range(iters):
            opt.zero_grad()
            L = loss_fn(params)
            L.backward()
            opt.step()
            if it % 100 == 0:
                log(f"   register it {it}: loss {float(L.detach()):.5f}")
    params = [p.detach() for p in params]
    out = world(params, g - g_c)
    iou1 = silhouette_iou(out, gen_faces.long(), K, E, masks, dist)
    info = {"init_yaw_deg": round(math.degrees(best[1]), 1), "scale": float(s0 * torch.exp(params[0])),
            "rotation_deg": round(math.degrees(float(params[1].norm())), 2),
            "translation": [round(float(x), 4) for x in params[2]],
            "iou_init_mean": round(float(np.mean(iou0)), 3), "iou_final_mean": round(float(np.mean(iou1)), 3)}
    log(f"   registration: IoU {info['iou_init_mean']} -> {info['iou_final_mean']} (yaw init {info['init_yaw_deg']}°)")
    return out.detach(), info


def recolour_from_photos(verts, faces, gen_colors, K, E, images, masks, dist=None, power: float = 2.0,
                         blend_weight: float = 0.3, vis_tol_frac: float = 0.004, abs_cos: bool = False,
                         harmonise: bool = True):
    """Photo colours where observed (mesh z-buffer visibility), generator colours elsewhere.

    a = clamp(Σ photo weights / blend_weight, 0, 1); colour = a·photo + (1−a)·generated.
    Returns (colours, observed fraction (a > 0.5)).
    """
    S, H, W = masks.shape
    faces = faces.long()
    normals = recon.vertex_normals(verts, faces)
    if ((verts - verts.mean(0)) * normals).sum() < 0:
        normals = -normals
    extent = float((verts.max(0).values - verts.min(0).values).norm())
    vis_depth = recon.mesh_depth_maps(verts, faces, K, E, (H, W), dist)
    centers = recon.camera_centers(E)
    acc = torch.zeros(len(verts), 3)
    wsum = torch.zeros(len(verts))
    for i in range(S):
        uv, z = recon.project(verts, K[i], E[i], recon._k1(dist, i))
        inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
        d = recon.sample(vis_depth[i][..., None], uv, "nearest")[:, 0]
        m = recon.sample(masks[i][..., None], uv, "nearest")[:, 0]
        vis = inside & (m >= 0.5) & ((z - d).abs() < vis_tol_frac * extent)
        cosv = (normals * F.normalize(centers[i] - verts, dim=-1)).sum(-1)
        # abs_cos: non-manifold generated meshes have inconsistently oriented normals (journal 0012)
        w = (cosv.abs() if abs_cos else cosv.clamp(min=0)) ** power * vis.float()
        acc += recon.sample(images[i], uv) * w[:, None]
        wsum += w
    photo = acc / wsum.clamp(min=1e-8)[:, None]
    a = (wsum / blend_weight).clamp(0, 1)[:, None]
    gen = gen_colors[:, :3].float()
    # Harmonise the generator's (unlit, saturated) colours with the photos: affine colour map fitted on
    # well-observed vertices, applied where the generated colour is used (unobserved regions, blends).
    obs = a[:, 0] > 0.9
    if harmonise and obs.sum() > 100:
        X = torch.cat([gen[obs], torch.ones(int(obs.sum()), 1)], 1)
        A = torch.linalg.lstsq(X, photo[obs]).solution                 # (4,3)
        gen = (torch.cat([gen, torch.ones(len(gen), 1)], 1) @ A).clamp(0, 1)
    colors = a * photo + (1 - a) * gen
    return colors.clamp(0, 1), float((a[:, 0] > 0.5).float().mean())
