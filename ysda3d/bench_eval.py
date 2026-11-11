"""Benchmark evaluation (05_benchmark_plan.md): the approaches built so far, scored on every example.

Each method turns a prepared example into a coloured mesh in the example's COLMAP frame, using only the
training photos. The mesh is then scored:
  - all examples: on held-out photos (silhouette IoU, full-mask and covered PSNR; as in journals 0007/0012);
  - synthetic examples, in addition:
    - the low held-out views (sides the training views saw only obliquely);
    - geometry against the ground-truth mesh after a camera-based alignment (Chamfer, F-score).
Expensive intermediates (reconstructions, repair, registration, recolouring) are cached per example,
keyed by input file hashes, parameters and the source of the ysda3d modules they use. So a re-run only
recomputes what a code change actually touches.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch

from . import bench, generated, meshfix, recon
from .sfm import _umeyama


def _surface_fn():
    from comfy_extras.nodes_gaussian_splat import _surface_nets
    return lambda v, l, vx, o: _surface_nets(v, l, vx, o, torch.device("cpu"))


def _centres(E):
    E = E.double()
    return torch.stack([-e[:3, :3].T @ e[:3, 3] for e in E]).numpy()


class Case:
    """One prepared example: photos, masks, COLMAP cameras, train/held-out split, generated mesh."""

    def __init__(self, ex):
        self.ex, self.name = ex, ex["name"]
        w = bench.work_dir(ex)
        self.sfm_path, self.da3_path = os.path.join(w, "sfm.npz"), os.path.join(w, "da3.npz")
        d = np.load(self.sfm_path)
        self.names = [str(n) for n in d["names"]]
        self.images, self.masks = torch.from_numpy(d["images"]), torch.from_numpy(d["masks"])
        self.depth, self.conf = torch.from_numpy(d["depth"]), torch.from_numpy(d["confidence"])
        self.K, self.E = torch.from_numpy(d["intrinsics"]).float(), torch.from_numpy(d["extrinsics"]).float()
        self.dist = torch.from_numpy(d["k1"]) if "k1" in d.files else None
        split = json.load(open(os.path.join(w, "split.json")))
        self.train = [self.names.index(n) for n in split["train"]]
        self.held = [self.names.index(n) for n in split["heldout"]]
        self.anchor = split["train"].index(split["generator_views"][0])
        self.meta = bench.meta(ex)
        gj = os.path.join(w, "generation.json")
        self.glb = json.load(open(gj))["glb"] if os.path.exists(gj) else None
        self.cache_dir = os.path.join(bench.ROOT, "cache", self.name)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.inputs = bench.file_hash(self.sfm_path, os.path.join(w, "split.json"))
        self.timings = {}

    def t(self, x):
        return x[self.train] if x is not None else None

    def cached(self, stage, deps, fn):
        key = hashlib.sha256(json.dumps(deps, sort_keys=True).encode()).hexdigest()[:16]
        path = os.path.join(self.cache_dir, f"{stage}_{key}.pt")
        if os.path.exists(path):
            return torch.load(path)
        t0 = time.time()
        out = fn()
        torch.save(out, path)
        self.timings[stage] = round(time.time() - t0, 1)
        return out

    # --- shared stages ----------------------------------------------------------------------------
    def recon_v12(self):
        def run():
            v, f, c, _ = recon.reconstruct(self.t(self.depth), self.t(self.K), self.t(self.E), self.t(self.images),
                                           self.t(self.masks), _surface_fn(), depth_tol=0.05, max_violations=1,
                                           conf=self.t(self.conf), dist=self.t(self.dist), visibility="mesh",
                                           color_power=2, log=lambda *a: None)
            return {"v": v, "f": f, "c": c}
        return self.cached("recon_v12", {"in": self.inputs, "code": bench.code_hash("recon")}, run)

    def recon_v0(self):
        """v0 on the same training photos, but with DA3's own cameras; mapped into the COLMAP frame for scoring."""
        def run():
            d = np.load(self.da3_path)
            names = [str(n) for n in d["names"]]
            idx = [names.index(self.names[i]) for i in self.train]
            H, W = d["depth"].shape[1:]
            K = recon.rescale_intrinsics(torch.from_numpy(d["intrinsics"]),
                                         recon.da3_processing_size(H, W, int(d["da3_res"])), (H, W))
            E = torch.from_numpy(d["extrinsics"]).float()
            pick = lambda a: torch.as_tensor(a)[idx]  # noqa: E731
            v, f, c, _ = recon.reconstruct(pick(d["depth"]), pick(K).float(), pick(E), pick(d["images"]), pick(d["masks"]),
                                           _surface_fn(), conf=pick(d["confidence"]), log=lambda *a: None)
            s, R, T = _umeyama(_centres(pick(E)), _centres(self.t(self.E)))
            v = torch.from_numpy((s * v.double().numpy() @ R.T + T).astype(np.float32))
            return {"v": v, "f": f, "c": c}
        return self.cached("recon_v0", {"in": self.inputs, "da3": bench.file_hash(self.da3_path),
                                        "code": bench.code_hash("recon")}, run)

    def floor(self):
        return self.cached("floor", {"in": self.inputs, "code": bench.code_hash("recon")},
                           lambda: recon.floor_plane(self.t(self.depth), self.t(self.K), self.t(self.E), self.t(self.masks),
                                                     dist=self.t(self.dist)))

    def generated(self, repaired: bool):
        """UniForge3D's mesh: repaired (narrow-band remesh, journal 0013) or raw + hidden-face cull (journal 0012)."""
        if self.glb is None:
            raise LookupError("generation pending")
        glb_hash = bench.file_hash(self.glb)

        def run():
            from comfy_api.latest._util.geometry_types import File3D
            from comfy_extras.nodes_mesh_io import Get3DComponents
            m = Get3DComponents.execute(File3D(self.glb)).result[0]
            v, f = m.vertices[0].float(), m.faces[0].long()
            c = m.vertex_colors[0].float()[:, :3] if m.vertex_colors is not None else torch.full_like(v, 0.7)
            if repaired:
                v, f, c = meshfix.remesh_narrow_band(v, f, c, res=512, log=lambda *a: None)
            else:
                v, f, used = meshfix.cull_hidden_faces(v, f, log=lambda *a: None)
                c = c[used]
            return {"v": v, "f": f, "c": c}
        return self.cached("repaired" if repaired else "culled", {"glb": glb_hash, "code": bench.code_hash("meshfix")}, run), glb_hash

    def registered(self, repaired: bool):
        g, glb_hash = self.generated(repaired)
        deps = {"in": self.inputs, "glb": glb_hash, "repaired": repaired,
                "code": bench.code_hash("generated", "recon", "meshfix", "diagnostics")}

        def run():
            wv, info = generated.register_generated_mesh(g["v"], g["f"], self.t(self.K), self.t(self.E), self.t(self.masks),
                                                         dist=self.t(self.dist), ref_verts=self.recon_v12()["v"],
                                                         floor=self.floor(), anchor=self.anchor, log=lambda *a: None)
            return {"v": wv, "info": info}
        return self.cached("register_" + ("repaired" if repaired else "raw"), deps, run), g, deps

    def recoloured(self, repaired: bool):
        r, g, deps = self.registered(repaired)

        def run():
            rc, obs, gh = generated.recolour_from_photos(r["v"], g["f"], g["c"], self.t(self.K), self.t(self.E),
                                                         self.t(self.images), self.t(self.masks), dist=self.t(self.dist),
                                                         abs_cos=True, return_harmonised=True)
            return {"photo": rc, "harmonised": gh, "observed": obs}
        return self.cached("recolour_" + ("repaired" if repaired else "raw"), deps, run), r, g

    def gt_in_sfm(self):
        """Ground-truth mesh (synthetic examples) moved into the COLMAP frame, plus the similarity used."""
        gt = np.load(os.path.join(bench.example_dir(self.ex), "gt", "mesh.npz"))
        cams = self.meta["cameras"]
        stem = [os.path.splitext(n)[0] for n in self.names]
        C_gt = np.stack([-np.array(cams[s]["R"]).T @ np.array(cams[s]["t"]) for s in stem])
        s, R, T = _umeyama(_centres(self.E), C_gt)                         # COLMAP -> GT
        v_sfm = ((gt["v"] - T) @ R) / s
        return gt, (s, R, T), torch.from_numpy(v_sfm.astype(np.float32))


# --- methods: example -> (verts, faces, colours) in the COLMAP frame, from training photos only ----------
def _m_recon_v0(c):
    m = c.recon_v0()
    return m["v"], m["f"], m["c"]


def _m_recon_v12(c):
    m = c.recon_v12()
    return m["v"], m["f"], m["c"]


def _m_gen_only(c):
    r, g, _ = c.registered(True)
    return r["v"], g["f"], g["c"]


def _m_gen_harmonised(c):
    rc, r, g = c.recoloured(True)
    return r["v"], g["f"], rc["harmonised"]


def _m_fused_v23(c):
    rc, r, g = c.recoloured(False)
    return r["v"], g["f"], rc["photo"]


def _m_fused_v231(c):
    rc, r, g = c.recoloured(True)
    return r["v"], g["f"], rc["photo"]


def subdivide(v: torch.Tensor, f: torch.Tensor, min_faces: int):
    """Midpoint subdivision (each triangle -> 4) until there are at least min_faces faces."""
    while len(f) < min_faces:
        e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]).sort(1).values
        uniq, inv = torch.unique(e, dim=0, return_inverse=True)
        mid = len(v) + inv.reshape(3, -1).T                                   # (F,3): midpoints of e01, e12, e20
        v = torch.cat([v, (v[uniq[:, 0]] + v[uniq[:, 1]]) / 2])
        a, b, c_ = f.T
        m01, m12, m20 = mid.T
        f = torch.cat([torch.stack(t, 1) for t in ((a, m01, m20), (m01, b, m12), (m20, m12, c_), (m01, m12, m20))])
    return v, f


def _m_oracle(c):
    """Synthetic only: the true mesh, coloured from the training photos like v2.3 (upper bound for colour)."""
    if c.meta.get("kind") != "synthetic":
        raise LookupError("needs ground truth")
    gt, _, v = c.gt_in_sfm()
    v, f = subdivide(v, torch.from_numpy(gt["f"]), 500_000)                  # vertex colours need dense vertices
    deps = {"in": c.inputs, "gt": bench.file_hash(os.path.join(bench.example_dir(c.ex), "gt", "mesh.npz")),
            "code": bench.code_hash("generated", "recon")}
    col = c.cached("oracle_colours", deps, lambda: generated.recolour_from_photos(
        v, f, torch.full_like(v, 0.5), c.t(c.K), c.t(c.E), c.t(c.images), c.t(c.masks), dist=c.t(c.dist))[0])
    return v, f, col


METHODS = {
    "recon_v0": ("v0: DA3 cameras, visual-hull carving, DA3-visibility colours", _m_recon_v0),
    "recon_v1_2": ("v1.2: COLMAP cameras, carving, mesh-visibility photo colours (reconstruction only)", _m_recon_v12),
    "gen_only": ("UniForge3D (repaired), registered, its own colours (generation only)", _m_gen_only),
    "gen_harmonised": ("UniForge3D (repaired), registered, own colours + global colour map", _m_gen_harmonised),
    "fused_v2_3": ("v2.3: raw UniForge3D mesh + cull, registered, photo colours", _m_fused_v23),
    "fused_v2_3_1": ("v2.3.1 (current): repaired mesh, registered, photo colours", _m_fused_v231),
    "oracle_gt_mesh": ("reference, synthetic only: true mesh + photo colours", _m_oracle),
}


# --- metrics ----------------------------------------------------------------------------------------
def render_mesh(v, f, col, K, E, k1: float, H: int, W: int, near: float = 1e-4):
    """Exact render of a vertex-coloured mesh (z-buffer, perspective-correct colours, SIMPLE_RADIAL k1 applied
    to the vertices; pixel centres at +0.5 as in COLMAP). Returns (image (H,W,3), coverage (H,W) bool).
    The diagnostics.* point-splat checks dilate silhouettes by ~2 px, which caps IoU near 0.96 for an exact
    shape and favours shrunken meshes, so the benchmark scores with this instead (journal 0016)."""
    from .synth import rasterize
    v, f, col = v.float(), f.long(), col.float()
    Xc = v @ E[:3, :3].T + E[:3, 3]
    z = Xc[:, 2]
    good = (z[f] > near).all(1)
    zs = z.clamp(min=near)
    x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
    if k1:
        r2 = x * x + y * y
        x, y = x * (1 + k1 * r2), y * (1 + k1 * r2)
    fg = f[good]
    fid, bary = rasterize(K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2], 1.0 / zs, fg, H, W)
    cov = fid >= 0
    tri = fg[fid[cov]]
    w = bary[cov] / zs[tri]
    w = w / w.sum(1, keepdim=True)
    img = torch.ones(H, W, 3)
    img[cov] = (col[tri] * w[..., None]).sum(1)
    return img, cov


def photo_metrics(c, v, f, col, views):
    """Mean over `views` of silhouette IoU, full-mask PSNR (uncovered mask pixels compared against white)
    and covered PSNR, from exact renders."""
    S, H, W = c.masks.shape
    ious, full, covd, cover = [], [], [], []
    for i in views:
        img, cov = render_mesh(v, f, col, c.K[i], c.E[i], recon._k1(c.dist, i), H, W)
        m = c.masks[i] >= 0.5
        ious.append(float((cov & m).sum()) / max(1, float((cov | m).sum())))
        se = ((img - c.images[i]) ** 2).sum(-1) / 3
        full.append(10 * np.log10(1 / max(float(se[m].mean()), 1e-10)))
        mc = m & cov
        covd.append(10 * np.log10(1 / max(float(se[mc].mean()), 1e-10)) if mc.any() else float("nan"))
        cover.append(float(mc.sum()) / max(1, float(m.sum())))
    return {"iou": float(np.mean(ious)), "psnr_full": float(np.mean(full)), "psnr_cov": float(np.nanmean(covd)),
            "coverage": float(np.mean(cover))}


def geometry_metrics(c, v_sfm, f, n: int = 150_000):
    from scipy.spatial import cKDTree
    gt, (s, R, T), _ = c.gt_in_sfm()
    vp = torch.from_numpy((s * v_sfm.double().numpy() @ R.T + T).astype(np.float32))
    P, _ = recon.surface_samples(vp, f.long(), n)
    G, _ = recon.surface_samples(torch.from_numpy(gt["v"]), torch.from_numpy(gt["f"]), n)
    P, G = P.numpy(), G.numpy()
    diag = float(np.linalg.norm(gt["v"].max(0) - gt["v"].min(0)))
    d_pg, d_gp = cKDTree(G).query(P, workers=-1)[0], cKDTree(P).query(G, workers=-1)[0]
    out = {"chamfer_pct": float((d_pg.mean() + d_gp.mean()) / 2 / diag * 100)}
    for tau in (0.01, 0.02):
        pr, rc = float((d_pg < tau * diag).mean()), float((d_gp < tau * diag).mean())
        out[f"fscore_{int(tau * 100)}pct"] = 2 * pr * rc / max(pr + rc, 1e-9)
        if tau == 0.02:
            out["precision_2pct"], out["recall_2pct"] = pr, rc
    return out


def evaluate_example(ex, methods):
    """Scores of each method on one example: {method: metrics or {"skipped": reason}}, plus diagnostics."""
    torch.set_grad_enabled(False)
    c = Case(ex)
    synthetic = c.meta.get("kind") == "synthetic"
    low = [i for i in c.held if os.path.splitext(c.names[i])[0] in c.meta.get("heldout_low", [])]
    interp = [i for i in c.held if i not in low]
    res = {}
    for name in methods:
        t0 = time.time()
        try:
            v, f, col = METHODS[name][1](c)
        except LookupError as e:                      # generation not ready / no ground truth
            res[name] = {"skipped": str(e)}
            continue
        def score():
            m = photo_metrics(c, v, f, col, interp)
            if synthetic and low:
                m.update({k + "_low": x for k, x in photo_metrics(c, v, f, col, low).items()})
            if synthetic:
                m.update(geometry_metrics(c, v, f))
            return m
        # scores are cached too: an unchanged method output costs a hash, not a re-render
        out_hash = hashlib.sha256(b"".join(np.ascontiguousarray(x.detach().cpu().numpy()).tobytes()
                                           for x in (v, f, col))).hexdigest()[:16]
        m = dict(c.cached("score_" + name, {"out": out_hash, "in": c.inputs, "views": [interp, low],
                                             "code": bench.code_hash("bench_eval", "recon", "synth")}, score))
        m["seconds"] = round(time.time() - t0, 1)
        res[name] = m
    diag = {"train_views": len(c.train), "heldout_views": len(interp), "heldout_low_views": len(low),
            "generated": c.glb is not None, "stage_seconds": c.timings}
    if synthetic:                                     # how good was the masking stage?
        from PIL import Image
        ious = []
        for i in c.held + c.train:
            stem = os.path.splitext(c.names[i])[0]
            gm = np.asarray(Image.open(os.path.join(bench.example_dir(ex), "gt", "masks", stem + ".png")), np.float32) / 255
            pm = c.masks[i].numpy()
            pm = pm[:gm.shape[0] * pm.shape[1] // gm.shape[1], :]                 # canvas -> photo aspect (top-left)
            gm = np.asarray(Image.fromarray((gm * 255).astype(np.uint8)).resize(pm.shape[::-1]), np.float32) / 255
            ious.append(float(((pm > 0.5) & (gm > 0.5)).sum() / max(((pm > 0.5) | (gm > 0.5)).sum(), 1)))
        diag["mask_iou_vs_gt"] = float(np.mean(ious))
    return {"example": ex["name"], "kind": c.meta.get("kind"), "methods": res, "diagnostics": diag}
