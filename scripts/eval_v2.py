"""v2.2 + v2.3 evaluation: register UniForge3D meshes into the COLMAP frame, recolour from photos,
and compare with v1.2 on held-out photos that NO method used.

  ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python scripts/eval_v2.py outputs/toy_sfm.npz outputs/v2eval
"""
import json
import os
import sys
import time

COMFY = os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
npz, prefix = sys.argv[1], sys.argv[2]
sys.argv = sys.argv[:1]
sys.path[:0] = [COMFY, REPO]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from comfy_api.latest._util.geometry_types import File3D  # noqa: E402
from comfy_extras.nodes_gaussian_splat import _surface_nets  # noqa: E402
from comfy_extras.nodes_mesh_io import Get3DComponents  # noqa: E402
from comfy_extras.nodes_save_3d import save_glb  # noqa: E402
from ysda3d import diagnostics, generated, meshfix, recon  # noqa: E402

torch.set_grad_enabled(False)
HELD = ["IMG_4506", "IMG_4510", "IMG_4516"]          # used by no method (not in UniForge3D K=4/K=8 inputs)
GEN = {"uf3d_k4": "outputs/uf3d/toy_k4.glb", "uf3d_k8": "outputs/uf3d/toy_k8.glb"}

d = np.load(npz)
names = [str(n) for n in d["names"]]
images, masks = torch.from_numpy(d["images"]), torch.from_numpy(d["masks"])
depth, conf = torch.from_numpy(d["depth"]), torch.from_numpy(d["confidence"])
K, E = torch.from_numpy(d["intrinsics"]).float(), torch.from_numpy(d["extrinsics"]).float()
dist = torch.from_numpy(d["k1"]) if "k1" in d.files else None
held = [i for i, n in enumerate(names) if any(n.startswith(h) for h in HELD)]
train = [i for i in range(len(names)) if i not in held]
anchor = next(i for i in train if names[i].startswith("IMG_4505"))
t_ = lambda x: x[train] if x is not None else None  # noqa: E731
print("held-out:", [names[i] for i in held], "| train:", len(train))


def evaluate(tag, verts, faces, colors):
    pm, tiles = diagnostics.photometric_check(verts, faces, colors, K, E, images, masks, held, dist=dist)
    _, ious = diagnostics.silhouette_check(verts, faces, K[held], E[held], masks[held],
                                           dist=dist[held] if dist is not None else None)
    r = {"psnr_full_mask": round(float(np.mean([x["psnr_full_mask"] for x in pm])), 2),
         "psnr_covered": round(float(np.mean([x["psnr"] for x in pm])), 2),
         "coverage": round(float(np.mean([x["coverage"] for x in pm])), 3),
         "silhouette_iou": round(float(np.mean(ious)), 3)}
    print(f"{tag:28s} PSNRfull {r['psnr_full_mask']:6.2f}  PSNRcov {r['psnr_covered']:6.2f}  "
          f"cov {r['coverage']:.3f}  IoU {r['silhouette_iou']:.3f}", flush=True)
    sheet = (torch.cat(list(tiles), 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(sheet).resize((sheet.shape[1] // 4, sheet.shape[0] // 4)).save(f"{prefix}_{tag}_heldout.jpg", quality=85)
    return r


results = {}
# --- v1.2 baseline (reconstruction only), rebuilt without the held-out photos ---------------------
v12, f12, c12, _ = recon.reconstruct(t_(depth), t_(K), t_(E), t_(images), t_(masks),
                                     lambda v, l, vx, o: _surface_nets(v, l, vx, o, torch.device("cpu")),
                                     depth_tol=0.05, max_violations=1, conf=t_(conf), dist=t_(dist),
                                     visibility="mesh", color_power=2, log=lambda *a: None)
results["v1.2"] = evaluate("v1.2", v12, f12, c12)
floor = recon.floor_plane(t_(depth), t_(K), t_(E), t_(masks), dist=t_(dist))
R_up = recon.upright_frame(t_(E), floor[0] if floor else None, train.index(anchor))
save_glb(recon.to_display_upright(v12, R_up, floor[1] if floor else None), f12, f"{prefix}_v1.2.glb", vertex_colors=c12)

# --- v2.2 / v2.3: generated meshes -----------------------------------------------------------------
for tag, path in GEN.items():
    m = Get3DComponents.execute(File3D(os.path.join(REPO, path))).result[0]
    gv, gf, gc = m.vertices[0].float(), m.faces[0].long(), m.vertex_colors[0].float()[:, :3]
    if os.environ.get("CULL", "1") == "1":
        # Keep the raw (correctly placed) surface but drop faces never frontmost from any viewpoint:
        # removes inner shells / buried fins that made per-vertex photo colours noisy (journal 0012).
        gv, gf, used = meshfix.cull_hidden_faces(gv, gf, log=print)
        gc = gc[used]
        tag = tag + "_cull"
    elif os.environ.get("REMESH", "0") == "1":
        # Clean watertight surface first: the raw dual-grid mesh has inner shells/fins that make
        # per-vertex photo colours noisy (journal 0012).
        gv, gf, gc = meshfix.remesh_watertight(gv, gf, lambda v, l, vx, o: _surface_nets(v, l, vx, o, torch.device("cpu")),
                                               gc, res=int(os.environ.get("REMESH_RES", "256")), log=print)
        tag = tag + "_remesh"
    t0 = time.time()
    wv, info = generated.register_generated_mesh(gv, gf, t_(K), t_(E), t_(masks), dist=t_(dist), ref_verts=v12,
                                                 floor=floor, anchor=train.index(anchor), log=print)
    info["seconds"] = round(time.time() - t0, 1)
    results[tag + " (v2.2 registered, generated colours)"] = {**evaluate(tag + "_gen", wv, gf, gc), "registration": info}
    t0 = time.time()
    rc, obs = generated.recolour_from_photos(wv, gf, gc, t_(K), t_(E), t_(images), t_(masks), dist=t_(dist),
                                             abs_cos=os.environ.get("CULL", "1") == "1")
    results[tag + " (v2.3 photo-recoloured)"] = {**evaluate(tag + "_photo", wv, gf, rc),
                                                 "observed_fraction": round(obs, 3), "seconds": round(time.time() - t0, 1)}
    disp = recon.to_display_upright(wv, R_up, floor[1] if floor else None)
    save_glb(disp, gf, f"{prefix}_{tag}_v2_3.glb", vertex_colors=rc)
json.dump({"held_out": [names[i] for i in held], "results": results}, open(prefix + "_results.json", "w"), indent=1)
