"""Generate ComfyUI UI-format workflows with per-stage diagnostics.

  python3 scripts/make_workflow.py workflows/photos_to_mesh_v0_diagnostics.ui.json        # v0
  python3 scripts/make_workflow.py workflows/photos_to_mesh_v1_colmap.ui.json --v1        # v1: + COLMAP cameras
  python3 scripts/make_workflow.py workflows/photos_to_mesh_v1_1.ui.json --v11            # v1.1: + mesh visibility
  python3 scripts/make_workflow.py workflows/photos_to_mesh_v1_2.ui.json --v12            # v1.2: + upright output
  python3 scripts/make_workflow.py workflows/photos_to_3d_v2_uniforge3d.ui.json --v2      # v2: UniForge3D
"""
import json
import sys
import uuid

nodes, links = [], []


# NOTE: node-to-node data (photo names, input folder) uses custom socket types (YSDA_NAMES, YSDA_DIR), never
# forceInput STRING: tools disagree on whether forceInput widgets take a widgets_values slot (journal 0011).
def node(nid, type_, pos, size, widgets=(), outputs=(), title=None):
    n = {"id": nid, "type": type_, "pos": pos, "size": size, "flags": {}, "order": nid, "mode": 0,
         "inputs": [], "outputs": [{"name": o[0], "type": o[1], "links": []} for o in outputs],
         "properties": {"Node name for S&R": type_}, "widgets_values": list(widgets)}
    if title:
        n["title"] = title
    nodes.append(n)
    return n


def link(src, out_idx, dst, in_name, type_):
    lid = len(links) + 1
    src["outputs"][out_idx]["links"].append(lid)
    dst["inputs"].append({"name": in_name, "type": type_, "link": lid})
    links.append([lid, src["id"], out_idx, dst["id"], len(dst["inputs"]) - 1, type_])


X0, X1, X2, X3 = 40, 420, 840, 1300

if "--v2" in sys.argv:
    # v2: photos -> masks -> RGBA views -> UniForge3D (Mac port, subprocess) -> GLB
    load = node(1, "YSDALoadImageFolder", [X0, 80], [320, 110], ["toy", 1024],
                [("images", "IMAGE"), ("valid_mask", "MASK"), ("names", "YSDA_NAMES")])
    bgm = node(2, "LoadBackgroundRemovalModel", [X0, 260], [320, 60], ["birefnet.safetensors"],
               [("BACKGROUND_REMOVAL", "BACKGROUND_REMOVAL")])
    rmbg = node(3, "RemoveBackground", [X1, 80], [300, 50], [], [("mask", "MASK")])
    link(bgm, 0, rmbg, "bg_removal_model", "BACKGROUND_REMOVAL"); link(load, 0, rmbg, "image", "IMAGE")
    ov = node(4, "YSDAMaskOverlay", [X1, 170], [300, 70], [], [("overlay", "IMAGE")])
    link(load, 0, ov, "images", "IMAGE"); link(rmbg, 0, ov, "mask", "MASK"); link(load, 1, ov, "valid_mask", "MASK")
    pv_mask = node(5, "PreviewImage", [X1, 280], [380, 420], [], [], title="A. Masks (all photos)")
    link(ov, 0, pv_mask, "images", "IMAGE")
    prep = node(6, "YSDAPrepareUniForge3DInputs", [X2, 80], [380, 200],
                ["toy", "IMG_4505,IMG_4509,IMG_4513,IMG_4518", 4],
                [("input_dir", "YSDA_DIR"), ("preview", "IMAGE"), ("selected", "STRING")],
                title="G0. Views for UniForge3D (first = anchor/front; empty list = K evenly spaced)")
    link(rmbg, 0, prep, "object_mask", "MASK"); link(load, 1, prep, "valid_mask", "MASK"); link(load, 2, prep, "names", "YSDA_NAMES")
    pv_views = node(7, "PreviewImage", [X2, 320], [420, 420], [], [], title="G0. Selected RGBA views")
    link(prep, 1, pv_views, "images", "IMAGE")
    txt_sel = node(8, "PreviewAny", [X2, 760], [420, 160], [], [("STRING", "STRING")], title="G0. Selected photos")
    link(prep, 2, txt_sel, "source", "*")
    uf = node(9, "YSDAUniForge3D", [X3, 80], [360, 130], ["full", 42, "fixed"],
              [("model_3d", "FILE_3D_GLB"), ("mesh", "MESH"), ("shape512_3d", "FILE_3D_GLB"), ("report", "STRING")],
              title="G. UniForge3D (Mac) — full ≈ 30 min, preview ≈ 5 min")
    link(prep, 0, uf, "input_dir", "YSDA_DIR")
    save = node(10, "SaveGLB", [X3 + 400, 80], [480, 520], ["ysda/v2_uniforge3d"], [], title="Final model (GLB)")
    link(uf, 0, save, "mesh", "FILE_3D_GLB")
    save512 = node(11, "SaveGLB", [X3 + 400, 640], [480, 460], ["ysda/v2_uniforge3d_shape512"], [],
                   title="Intermediate: shape-512 geometry")
    link(uf, 2, save512, "mesh", "FILE_3D_GLB")
    txt = node(12, "PreviewAny", [X3, 260], [360, 520], [], [("STRING", "STRING")], title="G. Report (stage timings)")
    link(uf, 3, txt, "source", "*")
    groups = [
        {"title": "0/A. Photos + masks", "bounding": [X0 - 20, 20, 820, 700], "color": "#3f789e"},
        {"title": "G0. Input adapter (RGBA at photo resolution)", "bounding": [X2 - 20, 20, 460, 920], "color": "#8A8"},
        {"title": "G. UniForge3D: joint reconstruction + generation", "bounding": [X3 - 20, 20, 920, 1100], "color": "#a1309b"},
    ]
    nodes.sort(key=lambda n: n["id"])
    wf = {"id": str(uuid.uuid4()), "revision": 0, "last_node_id": max(n["id"] for n in nodes),
          "last_link_id": len(links), "nodes": nodes, "links": links, "groups": groups, "config": {}, "extra": {},
          "version": 0.4}
    json.dump(wf, open(sys.argv[1], "w"), indent=1)
    print("wrote", sys.argv[1], len(nodes), "nodes", len(links), "links")
    sys.exit(0)
# Stage 0: inputs
load = node(1, "YSDALoadImageFolder", [X0, 80], [320, 110], ["toy", 1024],
            [("images", "IMAGE"), ("valid_mask", "MASK"), ("names", "YSDA_NAMES")])
bgm = node(2, "LoadBackgroundRemovalModel", [X0, 260], [320, 60], ["birefnet.safetensors"],
           [("BACKGROUND_REMOVAL", "BACKGROUND_REMOVAL")])
da3m = node(3, "LoadDA3Model", [X0, 380], [320, 90], ["depth_anything_3_base.safetensors", "fp32"],
            [("DA3_MODEL", "DA3_MODEL")])
# Stage A: masks
rmbg = node(4, "RemoveBackground", [X1, 80], [300, 50], [], [("mask", "MASK")])
link(bgm, 0, rmbg, "bg_removal_model", "BACKGROUND_REMOVAL"); link(load, 0, rmbg, "image", "IMAGE")
ov = node(5, "YSDAMaskOverlay", [X1, 170], [300, 70], [], [("overlay", "IMAGE")])
link(load, 0, ov, "images", "IMAGE"); link(rmbg, 0, ov, "mask", "MASK"); link(load, 1, ov, "valid_mask", "MASK")
pv_mask = node(6, "PreviewImage", [X1, 280], [380, 420], [], [], title="A. Masks (yellow = outline)")
link(ov, 0, pv_mask, "images", "IMAGE")
# Stage B: cameras + depth
da3 = node(7, "YSDADepthAnything3Multiview", [X2, 80], [320, 110], [504, "cam_dec"], [("da3_geometry", "DA3_GEOMETRY")])
link(da3m, 0, da3, "da3_model", "DA3_MODEL"); link(load, 0, da3, "images", "IMAGE")
dprev = node(8, "YSDADepthPreview", [X2, 230], [320, 70], [], [("depth", "IMAGE"), ("confidence", "IMAGE")])
link(da3, 0, dprev, "da3_geometry", "DA3_GEOMETRY"); link(rmbg, 0, dprev, "object_mask", "MASK")
pv_depth = node(9, "PreviewImage", [X2, 340], [400, 380], [], [], title="B1. DA3 depth (red = near)")
link(dprev, 0, pv_depth, "images", "IMAGE")
pv_conf = node(10, "PreviewImage", [X2, 740], [400, 380], [], [], title="B2. DA3 confidence (bright = confident)")
link(dprev, 1, pv_conf, "images", "IMAGE")
cons = node(11, "YSDACrossViewConsistency", [X2, 1140], [320, 100], [0.02],
            [("per_view", "IMAGE"), ("stats", "STRING"), ("point_cloud", "FILE_3D_PLY")])
link(da3, 0, cons, "da3_geometry", "DA3_GEOMETRY"); link(rmbg, 0, cons, "object_mask", "MASK")
link(load, 1, cons, "valid_mask", "MASK")
pv_cons = node(12, "PreviewImage", [X2, 1280], [600, 420], [], [],
               title="B3. Cross-view consistency: photo | other views here | depth disagreement (blue ok, red bad)")
link(cons, 0, pv_cons, "images", "IMAGE")
txt_cons = node(13, "PreviewAny", [X2 + 620, 1280], [320, 420], [], [("STRING", "STRING")], title="B3. Consistency stats")
link(cons, 1, txt_cons, "source", "*")
save_pc = node(14, "SaveGLB", [X2 + 620, 760], [360, 480], ["ysda/diagnostics/fused_points"], [],
               title="B4. Fused DA3 point cloud (red blobs = cameras)")
link(cons, 2, save_pc, "mesh", "FILE_3D_PLY")
V12 = "--v12" in sys.argv
V11 = "--v11" in sys.argv or V12
V1 = "--v1" in sys.argv or V11
geo_src, geo_out, mask_src, mask_out, valid_src, valid_out = da3, 0, rmbg, 0, load, 1
if V1:
    # Stage B': COLMAP refinement, then the same consistency check on the refined cameras.
    XR = X3 - 40
    ref = node(20, "YSDARefineCamerasCOLMAP", [XR, 80], [340, 260], ["toy", 1024, 1600, 200, True],
               [("da3_geometry", "DA3_GEOMETRY"), ("object_mask", "MASK"), ("valid_mask", "MASK"), ("report", "STRING")])
    link(da3, 0, ref, "da3_geometry", "DA3_GEOMETRY"); link(rmbg, 0, ref, "object_mask", "MASK")
    link(load, 1, ref, "valid_mask", "MASK"); link(load, 2, ref, "names", "YSDA_NAMES")
    txt_ref = node(21, "PreviewAny", [XR, 380], [340, 420], [], [("STRING", "STRING")], title="B'. COLMAP report")
    link(ref, 3, txt_ref, "source", "*")
    cons2 = node(22, "YSDACrossViewConsistency", [XR, 820], [340, 100], [0.02],
                 [("per_view", "IMAGE"), ("stats", "STRING"), ("point_cloud", "FILE_3D_PLY")])
    link(ref, 0, cons2, "da3_geometry", "DA3_GEOMETRY"); link(ref, 1, cons2, "object_mask", "MASK")
    link(ref, 2, cons2, "valid_mask", "MASK")
    pv_cons2 = node(23, "PreviewImage", [XR, 960], [600, 420], [], [],
                    title="B'3. Consistency AFTER COLMAP (compare with B3)")
    link(cons2, 0, pv_cons2, "images", "IMAGE")
    txt_cons2 = node(24, "PreviewAny", [XR + 620, 960], [320, 420], [], [("STRING", "STRING")],
                     title="B'3. Consistency stats AFTER COLMAP")
    link(cons2, 1, txt_cons2, "source", "*")
    save_pc2 = node(25, "SaveGLB", [XR + 360, 380], [360, 460], ["ysda/diagnostics/fused_points_colmap"], [],
                    title="B'4. Fused point cloud, COLMAP cameras")
    link(cons2, 2, save_pc2, "mesh", "FILE_3D_PLY")
    geo_src, geo_out, mask_src, mask_out, valid_src, valid_out = ref, 0, ref, 1, ref, 2
    X3 = X3 + 760
# Stage C/D: reconstruction
rec = node(15, "YSDAReconstructMeshV12" if V12 else "YSDAReconstructMeshV11" if V11 else "YSDAReconstructMesh",
           [X3 + 400, 80], [320, 330 if V12 else 260 if V11 else 180],
           [192, 0.05, 1, "mesh", 2.0, 0, True, "carve", False] if V12 else [192, 0.05, 1, "mesh", 2.0, 0] if V11 else [192, 0.02, 1],
           [("mesh", "MESH"), ("preview", "IMAGE"), ("report", "STRING"), ("silhouettes", "IMAGE")])
link(geo_src, geo_out, rec, "da3_geometry", "DA3_GEOMETRY"); link(mask_src, mask_out, rec, "object_mask", "MASK")
link(valid_src, valid_out, rec, "valid_mask", "MASK")
pv_sil = node(16, "PreviewImage", [X3 + 400, 300], [400, 380], [], [],
              title="C. Carving vs masks (white ok, green = carved away, magenta = extra)")
link(rec, 3, pv_sil, "images", "IMAGE")
pv_cmp = node(17, "PreviewImage", [X3 + 820, 300], [700, 380], [], [], title="D. Photos (top) vs final model (bottom)")
link(rec, 1, pv_cmp, "images", "IMAGE")
txt_rec = node(18, "PreviewAny", [X3 + 400, 700], [400, 360], [], [("STRING", "STRING")], title="C. Reconstruction report")
link(rec, 2, txt_rec, "source", "*")
save = node(19, "SaveGLB", [X3 + 820, 700], [460, 480], ["ysda/v1_2" if V12 else "ysda/v1_1" if V11 else "ysda/v1" if V1 else "ysda/v0"], [], title="Final model (GLB)")
link(rec, 0, save, "mesh", "MESH")

groups = [
    {"title": "0. Inputs", "bounding": [X0 - 20, 20, 360, 480], "color": "#3f789e"},
    {"title": "A. Background removal", "bounding": [X1 - 20, 20, 420, 700], "color": "#8A8"},
    {"title": "B. Depth Anything 3: cameras + depth", "bounding": [X2 - 20, 20, 1000, 1700], "color": "#b58b2a"},
    {"title": "C/D. Surface + colours", "bounding": [X3 + 380, 20, 1160, 1180], "color": "#a1309b"},
]
if V1:
    groups.append({"title": "B'. COLMAP camera refinement (v1)", "bounding": [X3 - 760 - 60, 20, 1000, 1400],
                   "color": "#2a6b8e"})
nodes.sort(key=lambda n: n["id"])
wf = {"id": str(uuid.uuid4()), "revision": 0, "last_node_id": max(n["id"] for n in nodes), "last_link_id": len(links),
      "nodes": nodes, "links": links, "groups": groups, "config": {}, "extra": {}, "version": 0.4}
json.dump(wf, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(nodes), "nodes", len(links), "links")
