"""v1 camera refinement: COLMAP structure-from-motion (pycolmap) on the full-resolution photos.

DA3 gives every view a pose in one pass, but its views agree only to ~1.8% of depth (journal 0004).
Here classical SfM recomputes the cameras: SIFT matching, incremental mapping, and bundle adjustment.
DA3's per-view depth maps are then rescaled to agree with the sparse SfM points they observe.

Coordinates:
- SfM works on EXIF-rotated photos resized to `sfm_max_side` (one camera per lens group).
- The rest of the pipeline uses the canvas from `images.load_folder` (resized to `max_side`,
  centred on a common padded canvas). `_to_canvas` maps SfM pixels/intrinsics into that canvas.
- Output poses are world→camera [R|t] in COLMAP's world frame (OpenCV axes, like DA3).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import torch
from PIL import Image, ImageOps


def _lens_key(img: Image.Image) -> str:
    exif = img.getexif()
    sub = exif.get_ifd(0x8769)
    focal = sub.get(0x920A)
    model = exif.get(0x0110, "cam")
    return f"{model}_{float(focal):.2f}mm".replace(" ", "_") if focal else "unknown_lens"


def prepare_sfm_images(paths: list[str], work_dir: str, sfm_max_side: int):
    """Write EXIF-rotated, resized copies grouped by lens into work_dir/images/<lens>/.

    Returns per-photo info: sfm relative name, original upright size, sfm size.
    """
    img_root = os.path.join(work_dir, "images")
    shutil.rmtree(img_root, ignore_errors=True)
    info = []
    for p in paths:
        raw = Image.open(p)
        lens = _lens_key(raw)
        im = ImageOps.exif_transpose(raw).convert("RGB")
        ow, oh = im.size
        s = min(1.0, sfm_max_side / max(ow, oh))
        if s < 1.0:
            im = im.resize((round(ow * s), round(oh * s)), Image.LANCZOS)
        # One COLMAP camera per folder, and a camera has one image size: group by lens AND size.
        group = f"{lens}_{im.size[0]}x{im.size[1]}"
        os.makedirs(os.path.join(img_root, group), exist_ok=True)
        rel = f"{group}/{os.path.splitext(os.path.basename(p))[0]}.jpg"
        im.save(os.path.join(img_root, rel), quality=95)
        info.append({"name": os.path.basename(p), "sfm_name": rel, "orig_wh": (ow, oh), "sfm_wh": im.size, "lens": lens})
    return img_root, info


def run_sfm(img_root: str, work_dir: str, camera_model: str = "SIMPLE_RADIAL", python: str | None = None, log=print):
    """Run ysda3d.sfm_worker in a subprocess (see its docstring for why) and load its results."""
    import json
    import subprocess
    import sys
    out = os.path.join(work_dir, "sfm_result.npz")
    for f in (out, out + ".json"):
        if os.path.exists(f):
            os.remove(f)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=repo)
    cmd = [python or sys.executable, "-m", "ysda3d.sfm_worker", img_root, work_dir, out, camera_model]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out):
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"COLMAP worker failed (exit {proc.returncode}):\n{tail}")
    meta = json.load(open(out + ".json"))
    log(f"COLMAP: reconstructions {meta['reconstructions']}, {meta['points3D']} points, "
        f"reprojection {meta['mean_reprojection_error_px']}")
    return dict(np.load(out)), meta


def _canvas_mapping(info_i, max_side: int, canvas_hw):
    """Affine map from SfM pixel coords to canvas pixel coords: x_c = a * x_s + b (per axis)."""
    ow, oh = info_i["orig_wh"]
    sw, sh = info_i["sfm_wh"]
    s_can = min(1.0, max_side / max(ow, oh))
    rw, rh = round(ow * s_can), round(oh * s_can)
    H, W = canvas_hw
    x0, y0 = (W - rw) // 2, (H - rh) // 2
    ax, ay = rw / sw, rh / sh
    # pixel-centre convention: (x + 0.5) * a - 0.5 + offset
    return (ax, 0.5 * ax - 0.5 + x0), (ay, 0.5 * ay - 0.5 + y0)


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, T) minimising ||dst - (s R src + T)||²."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def refine_geometry(geom: dict, names: list[str], paths: list[str], work_dir: str, max_side: int,
                    sfm_max_side: int = 1600, masks: torch.Tensor | None = None, near_frac: float = 0.08,
                    drop_unregistered: bool = True, min_points: int = 200,
                    max_rot_disagreement_deg: float = 10.0, python: str | None = None, log=print):
    """Return a copy of DA3 geometry with SfM cameras and per-view rescaled depth, plus a report.

    The per-view depth scale is fitted on SfM points near the object (inside the object mask
    dilated by near_frac of the image size): DA3's depth error is not a single global scale, so
    fitting on distant background points gives a wrong scale for the object (journal 0005).
    """
    depth = geom["depth"].float().cpu()
    S, H, W = depth.shape
    K_da3 = geom["ysda_K_canvas"]  # DA3 intrinsics already rescaled to the canvas (set by caller)
    E_da3 = geom["extrinsics"][0].float().cpu()

    img_root, info = prepare_sfm_images(paths, work_dir, sfm_max_side)
    res, meta = run_sfm(img_root, work_dir, python=python, log=log)
    near = None
    if masks is not None:
        k = max(3, int(near_frac * max(H, W)) | 1)
        near = (torch.nn.functional.max_pool2d((masks >= 0.5).float()[:, None], k, 1, k // 2)[:, 0] > 0).numpy()
    idx_of = {str(n): k for k, n in enumerate(res["names"])}

    K_out = K_da3.clone()
    E_out = torch.zeros(S, 3, 4)
    k1_out = torch.zeros(S)
    registered = []
    scales, spreads, n_pts, n_obs = [], [], [], []
    for i in range(S):
        k = idx_of.get(info[i]["sfm_name"])
        if k is None:
            registered.append(False)
            continue
        registered.append(True)
        Ks = res["K"][k]
        (ax, bx), (ay, by) = _canvas_mapping(info[i], max_side, (H, W))
        Kc = Ks.copy()
        Kc[0, 0] *= ax; Kc[0, 2] = ax * Ks[0, 2] + bx
        Kc[1, 1] *= ay; Kc[1, 2] = ay * Ks[1, 2] + by
        K_out[i] = torch.from_numpy(Kc).float()
        M = res["M"][k]
        E_out[i] = torch.from_numpy(M).float()
        params = meta["cameras"][info[i]["sfm_name"]]["params"]
        if "RADIAL" in meta["cameras"][info[i]["sfm_name"]]["model"]:
            k1_out[i] = params[3]   # SIMPLE_RADIAL: f, cx, cy, k (normalised coords: unchanged by resizing)

        # Per-view depth scale: SfM point depth / DA3 depth at the observing pixels.
        sel = res["obs_img"] == k
        n_obs.append(int(sel.sum()))
        X, xy = res["obs_xyz"][sel], res["obs_xy"][sel]
        z = (X @ M[:, :3].T + M[:, 3])[:, 2]
        u = np.round(ax * xy[:, 0] + bx).astype(int)
        v = np.round(ay * xy[:, 1] + by).astype(int)
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z = u[ok], v[ok], z[ok]
        if near is not None:
            close = near[i][v, u]
            if close.sum() >= 20:
                u, v, z = u[close], v[close], z[close]
        dd = depth[i].numpy()[v, u]
        r = z[dd > 0] / dd[dd > 0]
        s_i = float(np.median(r)) if len(r) >= 20 else float("nan")
        scales.append(s_i)
        spreads.append(float(np.median(np.abs(r / s_i - 1))) if len(r) >= 20 else float("nan"))
        n_pts.append(int(len(r)))

    # Sanity-check registrations against DA3: COLMAP can register a photo "consistently" from a few
    # background matches yet place it wrongly (journal 0005: IMG_4522/4523 with <150 points).
    # Align DA3 → SfM on well-supported views, then reject views whose rotation disagrees.
    reg = np.array(registered)
    pts_of = {}
    it = iter(n_obs)
    for i in range(S):
        if registered[i]:
            pts_of[i] = next(it)
    E_np = E_out.numpy()
    R_da3 = E_da3[:, :3, :3].numpy()
    C_da3_all = (-(E_da3[:, :3, :3].transpose(1, 2) @ E_da3[:, :3, 3:])[..., 0]).numpy()
    C_sfm_all = np.stack([-(E_np[i, :, :3].T @ E_np[i, :, 3]) for i in range(S)])
    strong = [i for i in pts_of if pts_of[i] >= min_points]
    rejected = {}
    if len(strong) >= 3:
        _, R_al, _ = _umeyama(C_da3_all[strong], C_sfm_all[strong])
        for i in pts_of:
            R_pred = R_da3[i] @ R_al.T            # DA3 orientation expressed in the SfM frame
            cosang = (np.trace(R_pred.T @ E_np[i, :, :3]) - 1) / 2
            ang = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
            if pts_of[i] < min_points or ang > max_rot_disagreement_deg:
                rejected[names[i]] = {"points": pts_of[i], "rot_vs_da3_deg": round(ang, 1)}
                registered[i] = False
                K_out[i] = K_da3[i]
                k1_out[i] = 0.0
    reg = np.array(registered)
    # Keep the per-view scale list aligned with the registered views.
    scales = [sc for i, sc in zip([j for j in range(S) if j in pts_of], scales) if registered[i]]
    if reg.sum() < 3:
        raise RuntimeError(f"only {int(reg.sum())} of {S} photos registered by COLMAP")
    good_scale = np.nanmedian(scales)
    # Unregistered views: carry DA3's pose into the SfM frame via a similarity fit on camera centres.
    C_da3 = (-(E_da3[:, :3, :3].transpose(1, 2) @ E_da3[:, :3, 3:])[..., 0]).numpy()
    E_np = E_out.numpy()
    C_sfm = np.stack([-(E_np[i, :, :3].T @ E_np[i, :, 3]) for i in range(S)])
    s_sim, R_sim, T_sim = _umeyama(C_da3[reg], C_sfm[reg])
    depth_out = depth.clone()
    si = iter(scales)
    for i in range(S):
        if registered[i]:
            s_i = next(si)
            depth_out[i] = depth[i] * (s_i if np.isfinite(s_i) else good_scale)
        else:
            Rd, td = E_da3[i, :, :3].numpy(), E_da3[i, :, 3].numpy()
            R_new = Rd @ R_sim.T
            t_new = -R_new @ T_sim + s_sim * td
            E_out[i] = torch.from_numpy(np.concatenate([R_new, t_new[:, None]], 1)).float()
            depth_out[i] = depth[i] * s_sim

    report = {
        "registered": int(reg.sum()), "views": S,
        "unregistered": [names[i] for i in range(S) if not registered[i]],
        "rejected_registrations": rejected,
        "sfm_points": meta["points3D"], "mean_reprojection_error_px": meta["mean_reprojection_error_px"],
        "cameras": {v["camera_id"]: {"model": v["model"], "params": [round(x, 4) for x in v["params"]]}
                    for v in meta["cameras"].values()},
        "per_view_depth_scale": [round(x, 4) if np.isfinite(x) else None for x in scales],
        "per_view_depth_residual_mad": [round(x, 4) if np.isfinite(x) else None for x in spreads],
        "per_view_sfm_points_near_object": n_pts,
        "per_view_sfm_observations": n_obs,
    }
    g = dict(geom)
    g["depth"] = depth_out
    g["extrinsics"] = E_out[None]
    g["intrinsics"] = K_out[None]
    keep = [i for i in range(S) if registered[i] or not drop_unregistered]
    if len(keep) < S:
        g = slice_views(g, keep)
    g["ysda_view_index"] = keep
    report["dropped_views"] = [names[i] for i in range(S) if i not in keep]
    g["ysda_intrinsics_canvas"] = True
    g["ysda_k1"] = k1_out
    g["ysda_sfm_report"] = report
    return g, report


def slice_views(g: dict, keep: list[int]) -> dict:
    """Keep only the given views in a DA3 geometry dict (per-view tensors and camera tensors)."""
    g = dict(g)
    idx = torch.tensor(keep)
    S = g["depth"].shape[0]
    for k, v in list(g.items()):
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == S and k not in ("extrinsics", "intrinsics"):
            g[k] = v[idx]
    for k in ("extrinsics", "intrinsics"):
        g[k] = g[k][:, idx]
    return g
