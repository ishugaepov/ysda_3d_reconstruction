"""ComfyUI node pack for the YSDA photos → 3D pipeline.

Thin wrappers around the repo's `ysda3d` package (pure torch/numpy). This folder is symlinked into
ComfyUI's custom_nodes/; restart ComfyUI after changing code.
"""
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import folder_paths  # noqa: E402
from comfy_api.latest._util.geometry_types import MESH  # noqa: E402

from ysda3d import images as yimages  # noqa: E402
from ysda3d import diagnostics, recon  # noqa: E402
from ysda3d.render import render_points  # noqa: E402

CATEGORY = "YSDA 3D"


def _input_subfolders():
    root = folder_paths.get_input_directory()
    names = sorted(d for d in os.listdir(root)
                   if not d.startswith(".") and os.path.isdir(os.path.join(root, d)))
    return names or ["(put a folder of photos into ComfyUI/input)"]


class YSDALoadImageFolder:
    """Load every photo in input/<folder>: EXIF rotation applied, resized, padded to one batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": (_input_subfolders(), {"tooltip": "Subfolder of ComfyUI's input directory"}),
            "max_side": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                 "tooltip": "Longest image side after resizing"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "YSDA_NAMES")
    RETURN_NAMES = ("images", "valid_mask", "names")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Loads an arbitrary set of photos (portrait and landscape mixed) as one batch. "
                   "valid_mask is 1 on real pixels and 0 on the padding.")

    @classmethod
    def IS_CHANGED(cls, folder, max_side):
        d = os.path.join(folder_paths.get_input_directory(), folder)
        return ";".join(f"{p}:{os.path.getmtime(p)}" for p in yimages.list_images(d))

    def load(self, folder, max_side):
        batch, valid, names = yimages.load_folder(
            os.path.join(folder_paths.get_input_directory(), folder), max_side)
        return torch.from_numpy(batch), torch.from_numpy(valid), "\n".join(names)


class YSDADepthAnything3Multiview:
    """Native Depth Anything 3 multi-view, with memory-efficient attention for many views on Mac."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "da3_model": ("DA3_MODEL", {"tooltip": "Load Depth Anything 3 (Small/Base; use fp32 on Mac)"}),
            "images": ("IMAGE",),
            "resolution": ("INT", {"default": 504, "min": 140, "max": 1512, "step": 14}),
            "pose_method": (["cam_dec", "ray_pose"], {"default": "cam_dec"}),
        }}

    RETURN_TYPES = ("DA3_GEOMETRY",)
    RETURN_NAMES = ("da3_geometry",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Same as Run Depth Anything 3 in multiview mode, but swaps ComfyUI's attention_basic "
                   "for sub-quadratic attention, which avoids out-of-memory crashes with ~20+ views on MPS.")

    def run(self, da3_model, images, resolution, pose_method):
        from comfy_extras.nodes_depth_anything_3 import DA3Inference
        from ysda3d.comfy_compat import memory_efficient_da3_attention
        mode = {"mode": "multiview", "ref_view_strategy": "saddle_balanced", "pose_method": pose_method}
        with memory_efficient_da3_attention():
            out = DA3Inference.execute(da3_model, images, resolution, "upper_bound_resize", mode)
        geom = dict(out.result[0] if hasattr(out, "result") else out[0])
        geom["ysda_process_res"] = resolution
        return (geom,)


class YSDAReconstructMesh:
    """DA3 cameras + depth + object masks → closed mesh with per-vertex colours from the photos."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "da3_geometry": ("DA3_GEOMETRY",),
            "object_mask": ("MASK", {"tooltip": "Per-photo foreground mask, e.g. from Remove Background"}),
            "voxel_resolution": ("INT", {"default": 192, "min": 64, "max": 384, "step": 16}),
            "depth_tolerance": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005,
                                          "tooltip": "Fraction of object size; larger keeps more volume"}),
            "max_violations": ("INT", {"default": 1, "min": 0, "max": 10,
                                       "tooltip": "Views allowed to disagree before a voxel is carved away"}),
        }, "optional": {
            "valid_mask": ("MASK", {"tooltip": "Padding mask from YSDA Load Image Folder"}),
        }}

    RETURN_TYPES = ("MESH", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("mesh", "preview", "report", "silhouettes")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("v0 reconstruction: voxel carving with masks + DA3 depth, surface nets, vertex colours "
                   "blended from the photos that see each vertex. Preview: photos (top) vs. renders (bottom).")

    def run(self, da3_geometry, object_mask, voxel_resolution, depth_tolerance, max_violations, valid_mask=None,
            visibility="da3", color_power=2.0, mask_erode_px=0, upright=False, surface_mode="carve",
            calibrate_depth=False):
        from comfy_extras.nodes_gaussian_splat import _surface_nets

        g = da3_geometry
        if g.get("mode") != "multiview" or "extrinsics" not in g:
            raise ValueError("needs multi-view DA3 geometry (camera poses); use YSDA Depth Anything 3 Multiview")
        depth, K, E, dist = _geometry_cameras(g)
        images = g["image"].float().cpu()
        S, H, W = depth.shape
        if object_mask.shape[0] != S:
            raise ValueError(f"{object_mask.shape[0]} masks for {S} views: after COLMAP refinement, use the "
                             "refine node's object_mask/valid_mask outputs (it drops unregistered photos)")
        masks = _fit_masks(object_mask, H, W, valid_mask)
        conf = g.get("confidence")
        conf = conf.float().cpu() if conf is not None else None

        def surface_fn(vol, level, voxel, origin):
            return _surface_nets(vol, level, voxel, origin, torch.device("cpu"))

        verts, faces, colors, report = recon.reconstruct(
            depth, K, E, images, masks, surface_fn, res=voxel_resolution, depth_tol=depth_tolerance,
            max_violations=max_violations, conf=conf, dist=dist, visibility=visibility,
            color_power=color_power, mask_erode_px=mask_erode_px, surface_mode=surface_mode,
            tsdf_calibrate=calibrate_depth, tsdf_bias_voxels=3.0 if surface_mode == "tsdf_in_hull" else 0.0)

        idx = sorted(set(np.linspace(0, S - 1, min(S, 6)).round().astype(int).tolist()))
        tiles = [torch.cat([images[i], torch.from_numpy(render_points(
            verts, colors, K[i], E[i], (H, W), k1=recon._k1(dist, i)))], 0) for i in idx]
        preview = torch.cat(tiles, 1)[None]

        sil, ious = diagnostics.silhouette_check(verts, faces, K, E, masks, dist=dist)
        report["silhouette_iou_mean"] = round(float(np.mean(ious)), 3)
        report["silhouette_iou_per_view"] = ious

        if upright:
            fp = recon.floor_plane(depth, K, E, masks, dist=dist)
            up, off = (fp[0], fp[1]) if fp is not None and fp[2] >= 0.2 else (None, None)
            R = recon.upright_frame(E, up)
            disp = recon.to_display_upright(verts, R, floor_offset=off)
            report["upright"] = {"source": "floor plane" if up is not None else "mean camera up (no floor found)",
                                 "floor_inlier_fraction": round(fp[2], 3) if fp is not None else None,
                                 "up_world": [round(float(x), 4) for x in R[1]]}
        else:
            disp = recon.to_display(verts)
        mesh = MESH(disp[None], faces[None], vertex_colors=colors[None])
        return mesh, preview, json.dumps(report, indent=1), sil


def _geometry_cameras(g):
    """depth (S,H,W), K on the canvas (S,3,3), E world→camera (S,3,4), k1 (S,) or None."""
    depth = g["depth"].float().cpu()
    S, H, W = depth.shape
    if g.get("ysda_intrinsics_canvas"):
        K = g["intrinsics"][0].float().cpu()
    else:
        res = int(g.get("ysda_process_res", 504))
        K = recon.rescale_intrinsics(g["intrinsics"][0].cpu(), recon.da3_processing_size(H, W, res), (H, W))
    dist = g.get("ysda_k1")
    return depth, K, g["extrinsics"][0].float().cpu(), (dist.float().cpu() if dist is not None else None)


def _fit_masks(masks, H, W, valid_mask=None):
    masks = masks.float().cpu()
    if masks.shape[-2:] != (H, W):
        masks = torch.nn.functional.interpolate(masks[:, None], size=(H, W), mode="bilinear")[:, 0]
    if valid_mask is not None:
        masks = masks * valid_mask.float().cpu()
    return masks


class YSDAReconstructMeshV11(YSDAReconstructMesh):
    """v1.1: colour visibility from the reconstructed mesh itself (z-buffer), looser depth carving."""

    @classmethod
    def INPUT_TYPES(cls):
        t = YSDAReconstructMesh.INPUT_TYPES()
        t["required"]["depth_tolerance"] = ("FLOAT", {"default": 0.05, "min": 0.0, "max": 10.0, "step": 0.005,
                                                       "tooltip": "Fraction of object size. 0.05 = v1.1 default; "
                                                                  ">= 1 disables the depth test (pure visual hull)"})
        t["required"]["visibility"] = (["mesh", "da3"], {"default": "mesh",
                                       "tooltip": "mesh: vertex visible if frontmost in the mesh's own z-buffer (v1.1); "
                                                  "da3: compare with DA3 depth (v0/v1)"})
        t["required"]["color_power"] = ("FLOAT", {"default": 2.0, "min": 0.0, "max": 64.0, "step": 0.5,
                                                  "tooltip": "View weight = cos(angle)^power. Very high values hurt (journal 0007)"})
        t["required"]["mask_erode_px"] = ("INT", {"default": 0, "min": 0, "max": 20,
                                                 "tooltip": "Ignore colour samples within this many px of the mask edge"})
        return t

    DESCRIPTION = ("v1.1: like v1, but colours use visibility from the mesh's own z-buffer (consistent with the "
                   "surface being coloured) and carving uses a looser DA3-depth tolerance. On the toy: held-out "
                   "full-mask PSNR 13.7 → 17.4 dB, silhouette IoU 0.88 → 0.92 (journal 0007).")


class YSDAReconstructMeshV12(YSDAReconstructMeshV11):
    """v1.2: v1.1 + upright output (floor plane → +Y, anchor photo → front). Surface options exposed."""

    @classmethod
    def INPUT_TYPES(cls):
        t = YSDAReconstructMeshV11.INPUT_TYPES()
        t["required"]["upright"] = ("BOOLEAN", {"default": True,
                                    "tooltip": "Rotate so the floor is horizontal (y=0, +Y up) and the first photo looks at the front. "
                                               "Falls back to the mean camera up-vector if no floor is found"})
        t["required"]["surface_mode"] = (["carve", "tsdf_in_hull"], {"default": "carve",
                                         "tooltip": "carve = v1.1 (best overall). tsdf_in_hull: depth fusion inside the mask hull "
                                                    "(sharper face, but narrower shape on the toy, journal 0008)"})
        t["required"]["calibrate_depth"] = ("BOOLEAN", {"default": False,
                                            "tooltip": "Rescale each DA3 depth map to the visual hull's depth first "
                                                       "(removes a ~4% bias, but ghosts concave regions; journal 0008)"})
        return t

    DESCRIPTION = ("v1.2: v1.1 reconstruction with upright output: a RANSAC floor plane under the object becomes "
                   "y=0 with +Y up, and the first photo's camera defines the front (+Z). The report shows the up-vector "
                   "source. TSDF/calibration options are available for experiments (journal 0008).")


class YSDAMaskOverlay:
    """Show the object masks on the photos (background dimmed, outline in yellow)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "mask": ("MASK",)},
                "optional": {"valid_mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/diagnostics"

    def run(self, images, mask, valid_mask=None):
        H, W = images.shape[1:3]
        return (diagnostics.mask_overlay(images.float().cpu(), _fit_masks(mask, H, W, valid_mask)),)


class YSDADepthPreview:
    """Stage-B view: DA3 depth (turbo colours, near = red) and confidence (bright = confident) per photo."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"da3_geometry": ("DA3_GEOMETRY",)},
                "optional": {"object_mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("depth", "confidence")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/diagnostics"

    def run(self, da3_geometry, object_mask=None):
        depth = da3_geometry["depth"].float().cpu()
        S, H, W = depth.shape
        m = _fit_masks(object_mask, H, W) >= 0.5 if object_mask is not None else torch.ones_like(depth, dtype=torch.bool)
        d_out, c_out = [], []
        conf = da3_geometry.get("confidence")
        for i in range(S):
            sel = depth[i][m[i]] if m[i].any() else depth[i].flatten()
            lo, hi = torch.quantile(sel, 0.02), torch.quantile(sel, 0.98)
            dimg = diagnostics._turbo(1 - (depth[i] - lo) / (hi - lo).clamp(min=1e-6))
            dimg[~m[i]] *= 0.3
            d_out.append(dimg)
            if conf is not None:
                c = conf[i].float().cpu()
                c = (c - torch.quantile(c, 0.02)) / (torch.quantile(c, 0.98) - torch.quantile(c, 0.02)).clamp(min=1e-6)
                c_out.append(c.clamp(0, 1)[..., None].repeat(1, 1, 3))
            else:
                c_out.append(torch.zeros(H, W, 3))
        return torch.stack(d_out), torch.stack(c_out)


class YSDACrossViewConsistency:
    """Stage-B check: do DA3's views agree? Projects every other view's 3D points into each view."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "da3_geometry": ("DA3_GEOMETRY",),
            "object_mask": ("MASK",),
            "rel_tolerance": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.2, "step": 0.001,
                                        "tooltip": "Depth disagreement (fraction of depth) shown as fully red at 3x this"}),
        }, "optional": {"valid_mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "STRING", "FILE_3D_PLY")
    RETURN_NAMES = ("per_view", "stats", "point_cloud")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/diagnostics"
    DESCRIPTION = ("Per view: [photo | other views' points rendered from this camera | depth disagreement: "
                   "blue = agree, red = disagree]. Point cloud: all views fused, cameras as red blobs.")

    def run(self, da3_geometry, object_mask, rel_tolerance, valid_mask=None):
        from comfy_api.latest._util.geometry_types import File3D
        depth, K, E, dist = _geometry_cameras(da3_geometry)
        S, H, W = depth.shape
        images = da3_geometry["image"].float().cpu()
        keep = _fit_masks(object_mask, H, W, valid_mask) >= 0.5
        tiles, stats, (pts, cols) = diagnostics.consistency_check(depth, K, E, images, keep, rel_tol=rel_tolerance,
                                                                  dist=dist)
        out_dir = os.path.join(folder_paths.get_output_directory(), "ysda", "diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "fused_points.ply")
        diagnostics.write_point_cloud_ply(path, pts, cols, recon.camera_centers(E))
        meds = [x["median_rel_depth_err"] for x in stats]
        summary = {"median_rel_depth_err_over_views": round(float(np.nanmedian(meds)), 4),
                   "worst_views": sorted(stats, key=lambda x: -x["median_rel_depth_err"])[:3],
                   "per_view": stats, "point_cloud": path}
        return tiles, json.dumps(summary, indent=1), File3D(path)


class YSDARefineCamerasCOLMAP:
    """v1: replace DA3's cameras with COLMAP structure-from-motion on the full-resolution photos."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "da3_geometry": ("DA3_GEOMETRY",),
            "object_mask": ("MASK",),
            "valid_mask": ("MASK",),
            "names": ("YSDA_NAMES", {"tooltip": "names output of YSDA Load Image Folder"}),
            "folder": (_input_subfolders(), {"tooltip": "Same folder as in YSDA Load Image Folder"}),
            "max_side": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                 "tooltip": "Must equal max_side of YSDA Load Image Folder"}),
            "sfm_max_side": ("INT", {"default": 1600, "min": 640, "max": 4096, "step": 64,
                                     "tooltip": "Resolution COLMAP matches features at"}),
            "min_points": ("INT", {"default": 200, "min": 0, "max": 5000,
                                   "tooltip": "Reject photos registered with fewer triangulated points"}),
            "drop_unregistered": ("BOOLEAN", {"default": True,
                                              "tooltip": "Drop photos COLMAP can't (reliably) register; "
                                                         "otherwise they keep DA3's pose"}),
        }}

    RETURN_TYPES = ("DA3_GEOMETRY", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("da3_geometry", "object_mask", "valid_mask", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("SIFT matching + incremental SfM + bundle adjustment (pycolmap, run in a subprocess), one "
                   "camera per lens and image size, radial distortion modelled. DA3 depth is rescaled per photo to "
                   "match the SfM points near the object. Masks are returned for the kept photos only.")

    def run(self, da3_geometry, object_mask, valid_mask, names, folder, max_side, sfm_max_side, min_points,
            drop_unregistered):
        from ysda3d import sfm
        name_list = [n for n in names.splitlines() if n.strip()]
        depth = da3_geometry["depth"]
        S, H, W = depth.shape
        if len(name_list) != S:
            raise ValueError(f"{len(name_list)} names for {S} views")
        res = int(da3_geometry.get("ysda_process_res", 504))
        g = dict(da3_geometry)
        g["ysda_K_canvas"] = recon.rescale_intrinsics(g["intrinsics"][0].cpu(), recon.da3_processing_size(H, W, res), (H, W))
        root = os.path.join(folder_paths.get_input_directory(), folder)
        work = os.path.join(folder_paths.get_temp_directory(), "ysda_sfm", folder)
        os.makedirs(work, exist_ok=True)
        masks = _fit_masks(object_mask, H, W, valid_mask)
        # The worker must run with this venv's python (pycolmap lives here; sys.executable may be the base one).
        venv_python = os.path.join(sys.prefix, "bin", "python3")
        out, report = sfm.refine_geometry(g, name_list, [os.path.join(root, n) for n in name_list], work, max_side,
                                          sfm_max_side=sfm_max_side, masks=masks, min_points=min_points,
                                          drop_unregistered=drop_unregistered,
                                          python=venv_python if os.path.exists(venv_python) else None)
        out.pop("ysda_K_canvas", None)
        keep = torch.tensor(out["ysda_view_index"])
        return out, object_mask[keep], valid_mask[keep], json.dumps(report, indent=1)


# ----------------------------------------------------------------------------------------------
# v2: UniForge3D (reconstruction + generation), run as a subprocess in its own venv (journal 0009)
# ----------------------------------------------------------------------------------------------
UNIFORGE_PYTHON = os.environ.get("YSDA_UNIFORGE_PYTHON", os.path.expanduser("~/src/uniforge3d-venv/bin/python"))
UNIFORGE_RUNNER = os.path.join(_REPO, "v2", "uniforge_mac", "run.py")


def _pick_views(names, views: str, k: int):
    """Explicit comma-separated name prefixes (first = anchor), else K evenly spaced in sorted order."""
    wanted = [v.strip() for v in views.replace("\n", ",").split(",") if v.strip()]
    if wanted:
        picks = []
        for w in wanted:
            hit = [i for i, n in enumerate(names) if n.startswith(w) or os.path.splitext(n)[0] == w]
            if not hit:
                raise ValueError(f"view {w!r} not found among {len(names)} photos")
            picks.append(hit[0])
        return picks
    k = max(1, min(k, len(names)))
    return sorted({int(round(x)) for x in np.linspace(0, len(names), k, endpoint=False)})


class YSDAPrepareUniForge3DInputs:
    """Write full-resolution RGBA views (photo + mask alpha) for UniForge3D; the first view is the anchor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "object_mask": ("MASK", {"tooltip": "Masks from Remove Background, on the loader's canvas"}),
            "valid_mask": ("MASK", {"tooltip": "valid_mask of YSDA Load Image Folder"}),
            "names": ("YSDA_NAMES", {"tooltip": "names output of YSDA Load Image Folder"}),
            "folder": (_input_subfolders(), {"tooltip": "Same folder as in YSDA Load Image Folder"}),
            "views": ("STRING", {"default": "", "multiline": False,
                                 "tooltip": "Comma-separated photo names/prefixes, first = anchor (front of the model). "
                                            "Empty = K photos evenly spaced in file order"}),
            "k": ("INT", {"default": 4, "min": 1, "max": 32, "tooltip": "Number of views when 'views' is empty "
                                                                         "(trained with 4; 8 looked more faithful, journal 0010)"}),
        }}

    RETURN_TYPES = ("YSDA_DIR", "IMAGE", "STRING")
    RETURN_NAMES = ("input_dir", "preview", "selected")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/v2"

    def run(self, object_mask, valid_mask, names, folder, views, k):
        name_list = [n for n in names.splitlines() if n.strip()]
        picks = _pick_views(name_list, views, k)
        out_dir = os.path.join(folder_paths.get_output_directory(), "ysda", "uniforge3d_inputs", folder)
        paths = yimages.rgba_views_from_masks(os.path.join(folder_paths.get_input_directory(), folder), name_list,
                                              object_mask.cpu().numpy(), valid_mask.cpu().numpy(), out_dir, picks)
        from PIL import Image
        tiles = []
        for p in paths:
            im = Image.open(p)
            im.thumbnail((512, 512))
            bg = Image.new("RGBA", (512, 512), (128, 128, 128, 255))
            bg.alpha_composite(im, ((512 - im.width) // 2, (512 - im.height) // 2))
            tiles.append(torch.from_numpy(np.asarray(bg.convert("RGB"), np.float32) / 255))
        return out_dir, torch.stack(tiles), "\n".join(name_list[i] for i in picks)


class YSDAUniForge3D:
    """UniForge3D (joint reconstruction + generation) on Apple Silicon, as a subprocess (journal 0009/0010)."""

    STAGE_WEIGHTS = {"load models": 110, "preprocess": 2, "dino": 5, "structure": 75, "shape512": 115,
                     "coords64": 3, "shape1024": 620, "texture": 900}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_dir": ("YSDA_DIR", {"tooltip": "Folder of RGBA PNGs (YSDA Prepare UniForge3D Inputs)"}),
            "mode": (["full", "preview (shape512, no colour)"], {"default": "full",
                     "tooltip": "full ≈ 30 min for 4–8 views; preview ≈ 5 min, geometry only"}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2 ** 31 - 1}),
        }}

    RETURN_TYPES = ("FILE_3D_GLB", "MESH", "FILE_3D_GLB", "STRING")
    RETURN_NAMES = ("model_3d", "mesh", "shape512_3d", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/v2"
    DESCRIPTION = ("Runs v2/uniforge_mac/run.py in ~/src/uniforge3d-venv (override with $YSDA_UNIFORGE_PYTHON). "
                   "Needs ~40 GB of unified memory while running; ComfyUI's loaded models are freed first. "
                   "A finished run is reused when the photos (exactly), masks (up to stray pixels), mode, seed and "
                   "port code all match (report: cached = true); change the seed to force a new generation.")

    def run(self, input_dir, mode, seed):
        import time
        from comfy_api.latest._util.geometry_types import File3D
        from comfy_extras.nodes_mesh_io import Get3DComponents
        from ysda3d import runcache

        preview = mode.startswith("preview")
        runs_root = os.path.join(folder_paths.get_output_directory(), "ysda", "uniforge3d")
        key = {"mode": mode, "seed": seed, "code": runcache.code_version(os.path.dirname(UNIFORGE_RUNNER))}
        views, alphas = runcache.fingerprint(input_dir)
        out_dir = runcache.find(runs_root, key, views, alphas)
        cached = out_dir is not None
        if not cached:
            out_dir = os.path.join(runs_root, time.strftime("%Y%m%d_%H%M%S"))
            self._generate(input_dir, out_dir, preview, seed)
            runcache.save(out_dir, key, views, alphas)
        st = json.load(open(os.path.join(out_dir, "status.json")))
        shape512 = st.get("shape512")
        final = shape512 if preview else st.get("output", os.path.join(out_dir, "model.glb"))
        mesh = Get3DComponents.execute(File3D(final)).result[0]
        report = {"mode": mode, "seed": seed, "input_dir": input_dir, "output_dir": out_dir, "cached": cached,
                  "stages": st.get("stages"), "total_seconds": st.get("total_seconds"),
                  "vertices": st.get("vertices"), "faces": st.get("faces")}
        return File3D(final), mesh, File3D(shape512), json.dumps(report, indent=1)

    def _generate(self, input_dir, out_dir, preview, seed):
        import subprocess
        import time
        import comfy.model_management as mm
        import comfy.utils

        if not os.path.exists(UNIFORGE_PYTHON):
            raise FileNotFoundError(f"UniForge3D venv python not found: {UNIFORGE_PYTHON} (see v2/uniforge_mac/README.md)")
        os.makedirs(out_dir, exist_ok=True)
        out_glb = os.path.join(out_dir, "model.glb")
        status_path = os.path.join(out_dir, "status.json")
        log_path = os.path.join(out_dir, "run.log")
        mm.unload_all_models()
        mm.soft_empty_cache()

        weights = dict(self.STAGE_WEIGHTS)
        if preview:
            for k in ("coords64", "shape1024", "texture"):
                weights.pop(k)
        total = sum(weights.values())
        pbar = comfy.utils.ProgressBar(total)
        cmd = [UNIFORGE_PYTHON, "-u", UNIFORGE_RUNNER, input_dir, out_glb, "--seed", str(seed),
               "--status-json", status_path, "--stop-after", "shape512" if preview else "all"]
        env = dict(os.environ, PYTHONPATH="", HF_HUB_OFFLINE="1")
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            done_w = 0
            while proc.poll() is None:
                time.sleep(2)
                try:
                    mm.throw_exception_if_processing_interrupted()
                except Exception:
                    proc.kill()
                    raise
                try:
                    st = json.load(open(status_path))
                    finished = {x["name"] for x in st.get("stages", [])}
                    done_w = weights["load models"] + sum(w for n, w in weights.items() if n in finished)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                pbar.update_absolute(min(done_w, total), total)
        tail = "\n".join(open(log_path).read().splitlines()[-25:])
        if proc.returncode != 0:
            raise RuntimeError(f"UniForge3D failed (exit {proc.returncode}); log {log_path}:\n{tail}")


class YSDARepairGeneratedMesh:
    """Hole-free, outward-wound, closed version of a generated mesh (narrow-band remesh, journal 0013)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh": ("MESH", {"tooltip": "mesh output of YSDA UniForge3D"}),
            "resolution": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64,
                                   "tooltip": "Grid cells along the longest side: 512 ≈ 1.9M faces, 768 ≈ 4.8M"}),
            "band": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.25,
                               "tooltip": "Offset of the new surface from the old one, in cells; "
                                          "holes up to ~2× this wide close"}),
            "seal": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.5,
                               "tooltip": "Extra cells for capping wider gaps into the interior; layers closer "
                                          "than ~2·(band + seal) cells merge"}),
            "close_bottom": ("BOOLEAN", {"default": True,
                                         "tooltip": "Cap openings that only face down (the never-photographed underside)"}),
        }}

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/v2"
    DESCRIPTION = ("The Mac port of UniForge3D lacks CuMesh's hole filling + narrow-band remesh, so its mesh has "
                   "cracks, flipped faces and an open bottom. This rebuilds the outer boundary of a thin band around "
                   "it (surface nets), closing holes and capping the underside; colours come from the nearest "
                   "original surface point.")

    def run(self, mesh, resolution, band, seal, close_bottom):
        import time
        from ysda3d import meshfix

        v, f = mesh.vertices[0].float().cpu(), mesh.faces[0].long().cpu()
        c = mesh.vertex_colors[0].float().cpu()[:, :3] if mesh.vertex_colors is not None else None
        t = time.time()
        nv, nf, nc = meshfix.remesh_narrow_band(v, f, c, res=resolution, band=band, seal=seal,
                                                close_bottom=close_bottom, log=lambda *a: None)
        if meshfix.signed_volume(nv, nf) < 0:                # closed surface: keep the faces wound outward
            nf = nf[:, [0, 2, 1]]
        report = {"input": meshfix.edge_stats(f), "repaired": meshfix.edge_stats(nf),
                  "seconds": round(time.time() - t, 1)}
        return (MESH(nv[None], nf[None], vertex_colors=nc[None] if nc is not None else None,
                     normals=recon.vertex_normals(nv, nf)[None]),       # smooth shading in the G preview
                json.dumps(report, indent=1))


class YSDAFitGeneratedMesh:
    """v2.2 + v2.3: register a generated mesh (UniForge3D) to the COLMAP cameras and colour it from the photos."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh": ("MESH", {"tooltip": "mesh output of YSDA Repair Generated Mesh (or YSDA UniForge3D)"}),
            "da3_geometry": ("DA3_GEOMETRY", {"tooltip": "from YSDA Refine Cameras (COLMAP)"}),
            "object_mask": ("MASK", {"tooltip": "object_mask output of YSDA Refine Cameras (COLMAP)"}),
            "valid_mask": ("MASK", {"tooltip": "valid_mask output of YSDA Refine Cameras (COLMAP)"}),
            "anchor_photo": ("STRING", {"default": "IMG_4505", "tooltip": "Name/prefix of the anchor (first) view given to UniForge3D"}),
            "names": ("YSDA_NAMES", {"tooltip": "names output of YSDA Load Image Folder"}),
            "colours": (["photos (v2.3)", "generated"], {"default": "photos (v2.3)"}),
            "register_iters": ("INT", {"default": 300, "min": 0, "max": 2000}),
        }}

    RETURN_TYPES = ("MESH", "IMAGE", "STRING")
    RETURN_NAMES = ("mesh", "preview", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/v2"
    DESCRIPTION = ("Culls hidden faces of a raw (open) generated mesh, fits a 7-DoF similarity to the masks in all COLMAP-"
                   "registered photos (initialised from a quick v1.2 reconstruction: floor-plane up, anchor = front), "
                   "then colours visible vertices from the photos; unobserved vertices keep the generated colours "
                   "harmonised to the photo palette (journal 0012). Preview: photos (top) vs result (bottom).")

    def run(self, mesh, da3_geometry, object_mask, valid_mask, anchor_photo, names, colours, register_iters):
        from comfy_extras.nodes_gaussian_splat import _surface_nets
        from ysda3d import generated, meshfix

        depth, K, E, dist = _geometry_cameras(da3_geometry)
        S, H, W = depth.shape
        images = da3_geometry["image"].float().cpu()
        masks = _fit_masks(object_mask, H, W, valid_mask)
        conf = da3_geometry.get("confidence")
        conf = conf.float().cpu() if conf is not None else None
        all_names = [n for n in names.splitlines() if n.strip()]
        kept = da3_geometry.get("ysda_view_index", list(range(S)))
        view_names = [all_names[i] for i in kept] if len(all_names) >= len(kept) else all_names
        anchor = next((i for i, n in enumerate(view_names) if n.startswith(anchor_photo)), 0)

        gv = mesh.vertices[0].float().cpu()
        gf = mesh.faces[0].long().cpu()
        gc = (mesh.vertex_colors[0].float().cpu()[:, :3] if mesh.vertex_colors is not None
              else torch.full((len(gv), 3), 0.7))
        report = {}
        clean = meshfix.edge_stats(gf)["open_edges"] == 0
        if not clean:
            # Raw generator output: drop inner shells / buried fins (journal 0012). A repaired, closed mesh has
            # none, and culling it would only cut holes in folds no test direction sees (journal 0013).
            gv, gf, used = meshfix.cull_hidden_faces(gv, gf, log=lambda *a: None)
            gc = gc[used]
            report["faces_after_cull"] = int(len(gf))
        sn = lambda v, l, vx, o: _surface_nets(v, l, vx, o, torch.device("cpu"))  # noqa: E731
        ref_v, _, _, _ = recon.reconstruct(depth, K, E, images, masks, sn, depth_tol=0.05, max_violations=1, conf=conf,
                                           dist=dist, visibility="mesh", log=lambda *a: None)
        floor = recon.floor_plane(depth, K, E, masks, dist=dist)
        wv, info = generated.register_generated_mesh(gv, gf, K, E, masks, dist=dist, ref_verts=ref_v, floor=floor,
                                                     anchor=anchor, iters=register_iters, log=lambda *a: None)
        report["registration"] = info
        if colours.startswith("photos"):
            cols, obs = generated.recolour_from_photos(wv, gf, gc, K, E, images, masks, dist=dist, abs_cos=True)
            report["observed_fraction"] = round(obs, 3)
        else:
            cols = gc
        R_up = recon.upright_frame(E, floor[0] if floor else None, anchor)
        disp = recon.to_display_upright(wv, R_up, floor[1] if floor else None)
        idx = sorted(set(np.linspace(0, S - 1, min(S, 6)).round().astype(int).tolist()))
        P, C = recon.surface_samples(wv, gf, 3_000_000, cols)
        tiles = [torch.cat([images[i], torch.from_numpy(render_points(P, C, K[i], E[i], (H, W), radius=1,
                                                                     k1=recon._k1(dist, i)))], 0) for i in idx]
        normals = None
        if clean:
            if meshfix.signed_volume(disp, gf) < 0:          # keep the winding outward after the transform
                gf = gf[:, [0, 2, 1]]
            normals = recon.vertex_normals(disp, gf)[None]    # smooth shading in viewers
        out = MESH(disp[None], gf[None], vertex_colors=cols[None], normals=normals)
        return out, torch.cat(tiles, 1)[None], json.dumps(report, indent=1)


NODE_CLASS_MAPPINGS = {
    "YSDARepairGeneratedMesh": YSDARepairGeneratedMesh,
    "YSDAFitGeneratedMesh": YSDAFitGeneratedMesh,
    "YSDAPrepareUniForge3DInputs": YSDAPrepareUniForge3DInputs,
    "YSDAUniForge3D": YSDAUniForge3D,
    "YSDAReconstructMeshV12": YSDAReconstructMeshV12,
    "YSDAReconstructMeshV11": YSDAReconstructMeshV11,
    "YSDARefineCamerasCOLMAP": YSDARefineCamerasCOLMAP,
    "YSDAMaskOverlay": YSDAMaskOverlay,
    "YSDADepthPreview": YSDADepthPreview,
    "YSDACrossViewConsistency": YSDACrossViewConsistency,
    "YSDALoadImageFolder": YSDALoadImageFolder,
    "YSDADepthAnything3Multiview": YSDADepthAnything3Multiview,
    "YSDAReconstructMesh": YSDAReconstructMesh,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YSDARepairGeneratedMesh": "YSDA Repair Generated Mesh (close holes)",
    "YSDAFitGeneratedMesh": "YSDA Fit Generated Mesh To Photos (v2.2 + v2.3)",
    "YSDAPrepareUniForge3DInputs": "YSDA Prepare UniForge3D Inputs (v2)",
    "YSDAUniForge3D": "YSDA UniForge3D (v2, Mac)",
    "YSDAReconstructMeshV12": "YSDA Reconstruct Mesh (v1.2, upright)",
    "YSDAReconstructMeshV11": "YSDA Reconstruct Mesh (v1.1)",
    "YSDARefineCamerasCOLMAP": "YSDA Refine Cameras (COLMAP)",
    "YSDAMaskOverlay": "YSDA Mask Overlay (diagnostic)",
    "YSDADepthPreview": "YSDA Depth & Confidence (diagnostic)",
    "YSDACrossViewConsistency": "YSDA Cross-View Consistency (diagnostic)",
    "YSDALoadImageFolder": "YSDA Load Image Folder",
    "YSDADepthAnything3Multiview": "YSDA Depth Anything 3 Multiview",
    "YSDAReconstructMesh": "YSDA Reconstruct Mesh (v0)",
}
