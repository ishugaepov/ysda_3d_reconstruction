"""Make UniForge3D (CUDA-only reference code) run on Apple Silicon without editing its source.

Call `apply()` BEFORE importing `uniforge3d`. It:
  1. puts pure-torch stand-ins for flash_attn / flex_gemm / o_voxel / cumesh / nvdiffrast first on sys.path;
  2. maps autocast("cuda") -> autocast("mps") and makes torch.cuda.set_device a no-op;
then `patch_pipeline()` (after import) replaces the two CUDA mesh post-processing steps:
  - Shape-512 post-process (CuMesh simplify/fill/narrow-band remesh) -> CPU decimation;
  - texture remesh + UV bake (CuMesh + nvdiffrast) -> CPU decimation + per-vertex PBR colours
    sampled from the texture voxels (base colour; metallic/roughness ignored).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UNIFORGE_SRC = os.environ.get("UNIFORGE3D_SRC", os.path.expanduser("~/src/UniForge3D"))


def apply():
    sys.path[:0] = [os.path.join(HERE, "shims"), UNIFORGE_SRC]
    import torch
    _Autocast = torch.autocast

    class MPSAutocast(_Autocast):
        def __init__(self, device_type, *a, **k):
            super().__init__("mps" if device_type == "cuda" else device_type, *a, **k)

    torch.autocast = MPSAutocast
    torch.cuda.set_device = lambda *a, **k: None

    # torch.segment_reduce has no MPS kernel; it's only used on small per-token tensors here.
    _segment_reduce = torch.segment_reduce

    def segment_reduce(data, reduce, *a, lengths=None, **k):
        if data.device.type == "mps":
            lengths = lengths.cpu() if isinstance(lengths, torch.Tensor) else lengths
            return _segment_reduce(data.cpu(), reduce, *a, lengths=lengths, **k).to(data.device)
        return _segment_reduce(data, reduce, *a, lengths=lengths, **k)

    torch.segment_reduce = segment_reduce
    if not torch.cuda.is_available():
        torch.cuda.empty_cache = lambda: torch.mps.empty_cache()


def _decimate(vertices, faces, target):
    import cumesh
    m = cumesh.CuMesh()
    m.init(vertices, faces)
    m.simplify(int(target))
    return m.read()


def patch_pipeline(log=print):
    import numpy as np
    import torch
    import trimesh
    import uniforge3d.pipelines.uniforge3d_image_to_3d as P
    from uniforge3d.representations.mesh.base import Mesh
    from flex_gemm.ops.grid_sample import grid_sample_3d

    Mesh.fill_holes = lambda self, *a, **k: None   # CuMesh boundary-loop hole filling: skipped

    def shape_postprocess(vertices, faces, *, grid_size, pre_simplify_target, decimation_target,
                          remesh_band, remesh_project):
        v, f = _decimate(vertices, faces, decimation_target)
        v = v.cpu().numpy().astype(np.float64)
        v[:, 1], v[:, 2] = v[:, 2].copy(), -v[:, 1].copy()           # model -> GLB display axes
        return trimesh.Trimesh(v, f.cpu().numpy(), process=False)

    def bake_vertex_colours(texture_config, mesh_config, mesh, pbr_voxel):
        v, f = _decimate(mesh.vertices, mesh.faces, texture_config.get("decimation_target", 1_000_000))
        R = int(mesh_config["grid_size"])
        feats, coords = pbr_voxel.feats.float(), pbr_voxel.coords
        grid = ((v.to(feats.device).float() + 0.5) * R)[None]          # aabb [-0.5,0.5]^3, voxel = 1/R
        attrs = grid_sample_3d(feats, coords, (1, feats.shape[1], R, R, R), grid, mode="trilinear")[0]
        rgb = (attrs[:, 0:3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        vn = v.cpu().numpy().astype(np.float64)
        vn[:, 1], vn[:, 2] = vn[:, 2].copy(), -vn[:, 1].copy()
        out = trimesh.Trimesh(vn, f.cpu().numpy(), process=False)
        out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], 1))
        return out

    P.official_shape_only_postprocess = shape_postprocess
    P.remesh_and_bake_texture = bake_vertex_colours

    class MacPipeline(P.UniForge3DImageTo3DPipeline):
        """Same pipeline on MPS; no background remover (inputs must be RGBA with real alpha).

        Use MacPipeline.from_pretrained("xibushu/UniForge3D", device="mps"). DINOv3 is read from
        $YSDA_DINO_PATH (a local Hugging Face folder) when set.
        """

        def __init__(self, config, *, device="mps", low_vram=True, cache_dir=None, local_files_only=False):
            if os.environ.get("YSDA_DINO_PATH"):
                config["external_models"]["dino"] = os.environ["YSDA_DINO_PATH"]
            # The VGGT conditioner enables autocast only on CUDA; on MPS it would feed fp32 inputs to
            # bf16 Linear layers and Metal's matmul asserts. Run it in fp32 instead (~4 GB).
            config["conditioner"]["vggt_feature_extractor"]["dtype"] = "float32"
            self.config = config
            self.device = torch.device(device)
            self.low_vram = low_vram
            t = time.time()
            with torch.device("cpu"):
                models = P.load_models(self.config, cache_dir=cache_dir, local_files_only=local_files_only)
            for k, v in models.items():
                setattr(self, k, v)
            log(f"models loaded in {time.time() - t:.0f}s")

            def no_rembg(image):
                raise ValueError("Mac port: give RGBA images with a real alpha mask (background removal is upstream)")
            self.remove_background = no_rembg

    return MacPipeline
