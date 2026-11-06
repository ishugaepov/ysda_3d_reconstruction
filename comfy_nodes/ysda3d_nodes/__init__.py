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

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
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

    def run(self, da3_geometry, object_mask, voxel_resolution, depth_tolerance, max_violations, valid_mask=None):
        from comfy_extras.nodes_gaussian_splat import _surface_nets

        g = da3_geometry
        if g.get("mode") != "multiview" or "extrinsics" not in g:
            raise ValueError("needs multi-view DA3 geometry (camera poses); use YSDA Depth Anything 3 Multiview")
        depth = g["depth"].float().cpu()
        images = g["image"].float().cpu()
        S, H, W = depth.shape
        masks = object_mask.float().cpu()
        if masks.shape[-2:] != (H, W):
            masks = torch.nn.functional.interpolate(masks[:, None], size=(H, W), mode="bilinear")[:, 0]
        if valid_mask is not None:
            masks = masks * valid_mask.float().cpu()
        res = int(g.get("ysda_process_res", 504))
        K = recon.rescale_intrinsics(g["intrinsics"][0].cpu(), recon.da3_processing_size(H, W, res), (H, W))
        E = g["extrinsics"][0].float().cpu()
        conf = g.get("confidence")
        conf = conf.float().cpu() if conf is not None else None

        def surface_fn(vol, level, voxel, origin):
            return _surface_nets(vol, level, voxel, origin, torch.device("cpu"))

        verts, faces, colors, report = recon.reconstruct(
            depth, K, E, images, masks, surface_fn, res=voxel_resolution, depth_tol=depth_tolerance,
            max_violations=max_violations, conf=conf)

        idx = sorted(set(np.linspace(0, S - 1, min(S, 6)).round().astype(int).tolist()))
        tiles = [torch.cat([images[i], torch.from_numpy(render_points(verts, colors, K[i], E[i], (H, W)))], 0)
                 for i in idx]
        preview = torch.cat(tiles, 1)[None]

        sil, ious = diagnostics.silhouette_check(verts, faces, K, E, masks)
        report["silhouette_iou_mean"] = round(float(np.mean(ious)), 3)
        report["silhouette_iou_per_view"] = ious

        mesh = MESH(recon.to_display(verts)[None], faces[None], vertex_colors=colors[None])
        return mesh, preview, json.dumps(report, indent=1), sil


def _geometry_cameras(g):
    depth = g["depth"].float().cpu()
    S, H, W = depth.shape
    res = int(g.get("ysda_process_res", 504))
    K = recon.rescale_intrinsics(g["intrinsics"][0].cpu(), recon.da3_processing_size(H, W, res), (H, W))
    return depth, K, g["extrinsics"][0].float().cpu()


def _fit_masks(masks, H, W, valid_mask=None):
    masks = masks.float().cpu()
    if masks.shape[-2:] != (H, W):
        masks = torch.nn.functional.interpolate(masks[:, None], size=(H, W), mode="bilinear")[:, 0]
    if valid_mask is not None:
        masks = masks * valid_mask.float().cpu()
    return masks


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
        depth, K, E = _geometry_cameras(da3_geometry)
        S, H, W = depth.shape
        images = da3_geometry["image"].float().cpu()
        keep = _fit_masks(object_mask, H, W, valid_mask) >= 0.5
        tiles, stats, (pts, cols) = diagnostics.consistency_check(depth, K, E, images, keep, rel_tol=rel_tolerance)
        out_dir = os.path.join(folder_paths.get_output_directory(), "ysda", "diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "fused_points.ply")
        diagnostics.write_point_cloud_ply(path, pts, cols, recon.camera_centers(E))
        meds = [x["median_rel_depth_err"] for x in stats]
        summary = {"median_rel_depth_err_over_views": round(float(np.nanmedian(meds)), 4),
                   "worst_views": sorted(stats, key=lambda x: -x["median_rel_depth_err"])[:3],
                   "per_view": stats, "point_cloud": path}
        return tiles, json.dumps(summary, indent=1), File3D(path)


NODE_CLASS_MAPPINGS = {
    "YSDAMaskOverlay": YSDAMaskOverlay,
    "YSDADepthPreview": YSDADepthPreview,
    "YSDACrossViewConsistency": YSDACrossViewConsistency,
    "YSDALoadImageFolder": YSDALoadImageFolder,
    "YSDADepthAnything3Multiview": YSDADepthAnything3Multiview,
    "YSDAReconstructMesh": YSDAReconstructMesh,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YSDAMaskOverlay": "YSDA Mask Overlay (diagnostic)",
    "YSDADepthPreview": "YSDA Depth & Confidence (diagnostic)",
    "YSDACrossViewConsistency": "YSDA Cross-View Consistency (diagnostic)",
    "YSDALoadImageFolder": "YSDA Load Image Folder",
    "YSDADepthAnything3Multiview": "YSDA Depth Anything 3 Multiview",
    "YSDAReconstructMesh": "YSDA Reconstruct Mesh (v0)",
}
