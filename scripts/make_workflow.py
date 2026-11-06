"""Generate the ComfyUI UI-format workflow for pipeline v0 + per-stage diagnostics.

  python3 scripts/make_workflow.py workflows/photos_to_mesh_v0_diagnostics.ui.json
"""
import json
import sys
import uuid

nodes, links = [], []


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
# Stage 0: inputs
load = node(1, "YSDALoadImageFolder", [X0, 80], [320, 110], ["toy", 1024],
            [("images", "IMAGE"), ("valid_mask", "MASK"), ("names", "STRING")])
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
# Stage C/D: reconstruction
rec = node(15, "YSDAReconstructMesh", [X3 + 400, 80], [320, 180], [192, 0.02, 1],
           [("mesh", "MESH"), ("preview", "IMAGE"), ("report", "STRING"), ("silhouettes", "IMAGE")])
link(da3, 0, rec, "da3_geometry", "DA3_GEOMETRY"); link(rmbg, 0, rec, "object_mask", "MASK")
link(load, 1, rec, "valid_mask", "MASK")
pv_sil = node(16, "PreviewImage", [X3 + 400, 300], [400, 380], [], [],
              title="C. Carving vs masks (white ok, green = carved away, magenta = extra)")
link(rec, 3, pv_sil, "images", "IMAGE")
pv_cmp = node(17, "PreviewImage", [X3 + 820, 300], [700, 380], [], [], title="D. Photos (top) vs final model (bottom)")
link(rec, 1, pv_cmp, "images", "IMAGE")
txt_rec = node(18, "PreviewAny", [X3 + 400, 700], [400, 360], [], [("STRING", "STRING")], title="C. Reconstruction report")
link(rec, 2, txt_rec, "source", "*")
save = node(19, "SaveGLB", [X3 + 820, 700], [460, 480], ["ysda/v0"], [], title="Final model (GLB)")
link(rec, 0, save, "mesh", "MESH")

groups = [
    {"title": "0. Inputs", "bounding": [X0 - 20, 20, 360, 480], "color": "#3f789e"},
    {"title": "A. Background removal", "bounding": [X1 - 20, 20, 420, 700], "color": "#8A8"},
    {"title": "B. Depth Anything 3: cameras + depth", "bounding": [X2 - 20, 20, 1000, 1700], "color": "#b58b2a"},
    {"title": "C/D. Surface + colours", "bounding": [X3 + 380, 20, 1160, 1180], "color": "#a1309b"},
]
wf = {"id": str(uuid.uuid4()), "revision": 0, "last_node_id": len(nodes), "last_link_id": len(links),
      "nodes": nodes, "links": links, "groups": groups, "config": {}, "extra": {}, "version": 0.4}
json.dump(wf, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(nodes), "nodes", len(links), "links")
