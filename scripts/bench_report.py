"""Benchmark report: input photos, held-out comparisons and every method's model, for a visual check.

  ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python -I scripts/bench_report.py [--examples a,b] [--workers 3]

Run after scripts/bench_run.py (method outputs come from its caches, so this mostly renders). Writes:
  $YSDA_BENCH_ROOT/report/index.html            open in a browser
  $YSDA_BENCH_ROOT/report/<example>/*.jpg       inputs.jpg, heldout.jpg (photo vs each method), models.jpg
  ~/ComfyUI-Shared/input/3d/ysda_bench/<example>/<method>.glb
      upright models; in ComfyUI they appear in the Load 3D node's file list (refresh with R)
Real examples use the pipeline's upright frame (floor plane up, anchor photo = front); synthetic ones use the
ground-truth frame, where gt_mesh.glb is the true object.
"""
import argparse
import html
import json
import multiprocessing
import os
import sys
import time

COMFY = os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLB_ROOT = os.path.expanduser("~/ComfyUI-Shared/input/3d/ysda_bench")
SHORT = {"recon_v0": "v0", "recon_v1_2": "v1.2 recon", "gen_only": "gen only", "gen_harmonised": "gen + colour map",
         "fused_v2_3": "v2.3", "fused_v2_3_1": "v2.3.1 (current)", "oracle_gt_mesh": "oracle (true mesh)"}


def _init(threads):
    sys.argv = sys.argv[:1]
    sys.path[:0] = [COMFY, REPO]
    import torch
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)


def _label(im, text, color=(0, 0, 0)):
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(255, 255, 255))
    d.text((4, 2), text, fill=color)
    return im


def _to_pil(img):
    import numpy as np
    from PIL import Image
    a = img.numpy() if hasattr(img, "numpy") else img
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def _orbit_views(v, size=420):
    """Cameras looking at the model centre: front, right, back (20° up) and top; display frame, +Y up."""
    import math
    import numpy as np
    import torch
    from ysda3d import synth
    lo, hi = v.min(0).values.double().numpy(), v.max(0).values.double().numpy()
    c, r = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2
    f = size / 2 / math.tan(math.radians(20))
    K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    d = r / math.sin(math.radians(20)) * 1.05
    cams = []
    for az, el in ((0, 20), (90, 20), (180, 20), (0, 85)):
        a, e = math.radians(az), math.radians(el)
        pos = c + d * np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        R, t = synth.look_at(pos, c, roll_deg=0.0)
        E = torch.eye(4)
        E[:3, :3], E[:3, 3] = torch.from_numpy(R).float(), torch.from_numpy(t).float()
        cams.append(E)
    return K, cams


def _work(ex, methods, out_root):
    import numpy as np
    import torch
    from PIL import Image
    from comfy_extras.nodes_save_3d import save_glb
    from ysda3d import bench, bench_eval, recon
    t0 = time.time()
    c = bench_eval.Case(ex)
    synthetic = c.meta.get("kind") == "synthetic"
    out = os.path.join(out_root, ex["name"])
    glb_dir = os.path.join(GLB_ROOT, ex["name"])
    os.makedirs(out, exist_ok=True)
    os.makedirs(glb_dir, exist_ok=True)
    S, H, W = c.masks.shape
    split = json.load(open(os.path.join(bench.work_dir(ex), "split.json")))

    # display frame shared by all methods of this example
    if synthetic:
        _, (s, R, T), _ = c.gt_in_sfm()
        to_disp = lambda v: torch.from_numpy((s * v.double().numpy() @ R.T + T).astype(np.float32))  # noqa: E731
    else:
        floor = c.floor()
        R_up = recon.upright_frame(c.t(c.E), floor[0] if floor else None, c.anchor)
        to_disp = lambda v: recon.to_display_upright(v, R_up, floor[1] if floor else None)  # noqa: E731

    # 1. inputs: every photo, held-out ones in red, generator inputs in blue
    tiles = []
    for i, n in enumerate(c.names):
        im = _to_pil(c.images[i])
        im.thumbnail((220, 220))
        role = "held out" if i in c.held else ("generator input" if n in split["generator_views"] else "train")
        col = {"held out": (200, 0, 0), "generator input": (0, 70, 200), "train": (0, 0, 0)}[role]
        tiles.append(_label(im, f"{os.path.splitext(n)[0]} · {role}", col))
    cols = 6
    th = max(t.height for t in tiles)
    sheet = Image.new("RGB", (220 * cols, th * ((len(tiles) + cols - 1) // cols)), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, (220 * (k % cols), th * (k // cols)))
    sheet.save(os.path.join(out, "inputs.jpg"), quality=85)

    # 2. method outputs: GLB + held-out renders + orbit views
    outputs, glbs = {}, {}
    for m in methods:
        try:
            v, f, col = bench_eval.METHODS[m][1](c)
        except LookupError:
            continue
        outputs[m] = (v, f.long(), col.float())
        path = os.path.join(glb_dir, f"{m}.glb")
        save_glb(to_disp(v), f.long(), path, vertex_colors=col.float().clamp(0, 1))
        glbs[m] = path
    if synthetic:
        gt = np.load(os.path.join(bench.example_dir(ex), "gt", "mesh.npz"))
        path = os.path.join(glb_dir, "gt_mesh.glb")
        save_glb(torch.from_numpy(gt["v"]), torch.from_numpy(gt["f"]), path)
        glbs["gt_mesh"] = path

    rows = []
    for i in c.held:
        ys, xs = torch.nonzero(c.masks[i] >= 0.5, as_tuple=True)
        if len(ys) == 0:
            continue
        m_ = int(0.12 * max(int(ys.max() - ys.min()), int(xs.max() - xs.min())))
        box = (max(0, int(xs.min()) - m_), max(0, int(ys.min()) - m_), min(W, int(xs.max()) + m_), min(H, int(ys.max()) + m_))
        row = [_label(_to_pil(c.images[i]).crop(box).resize((260, 260)), f"photo {os.path.splitext(c.names[i])[0]}", (200, 0, 0))]
        for m, (v, f, col) in outputs.items():
            img, cov = bench_eval.render_mesh(v, f, col, c.K[i], c.E[i], recon._k1(c.dist, i), H, W)
            mk = c.masks[i] >= 0.5
            se = ((img - c.images[i]) ** 2).sum(-1) / 3
            psnr = 10 * np.log10(1 / max(float(se[mk].mean()), 1e-10))
            row.append(_label(_to_pil(img).crop(box).resize((260, 260)), f"{SHORT.get(m, m)} · {psnr:.1f} dB"))
        rows.append(row)
    if rows:
        sheet = Image.new("RGB", (260 * max(len(r) for r in rows), 260 * len(rows)), "white")
        for y, row in enumerate(rows):
            for x, t in enumerate(row):
                sheet.paste(t, (260 * x, 260 * y))
        sheet.save(os.path.join(out, "heldout.jpg"), quality=85)

    # orbit views of every model in the display frame (front, right, back, top)
    model_rows = []
    named = list(outputs.items())
    if synthetic:
        gtv, gtf = torch.from_numpy(gt["v"]), torch.from_numpy(gt["f"])
        gtv_sfm = c.gt_in_sfm()[2]
        named.append(("gt_mesh", (gtv_sfm, gtf, torch.full_like(gtv_sfm, 0.7))))
    for m, (v, f, col) in named:
        vd = to_disp(v)
        K, cams = _orbit_views(vd)
        row = []
        normals = recon.vertex_normals(vd, f) if m == "gt_mesh" else None
        for k, E in enumerate(cams):
            if normals is not None:          # uncoloured true mesh: shade by orientation (headlight) to show its shape
                cam = -E[:3, :3].T @ E[:3, 3]
                d = torch.nn.functional.normalize(cam[None] - vd, dim=-1)
                col = (0.25 + 0.65 * (normals * d).sum(-1).abs())[:, None].expand(-1, 3) * torch.tensor([0.85, 0.85, 0.9])
            img, _ = bench_eval.render_mesh(vd, f, col, K, E, 0.0, 420, 420)
            row.append(_label(_to_pil(img), f"{SHORT.get(m, m)} · {['front', 'right', 'back', 'top'][k]}"))
        model_rows.append(row)
    sheet = Image.new("RGB", (420 * 4, 420 * len(model_rows)), "white")
    for y, row in enumerate(model_rows):
        for x, t in enumerate(row):
            sheet.paste(t, (420 * x, 420 * y))
    sheet.save(os.path.join(out, "models.jpg"), quality=82)
    return {"example": ex["name"], "kind": c.meta.get("kind"), "glbs": glbs, "photos": bench.photos_dir(ex),
            "seconds": round(time.time() - t0, 1)}


def write_html(out_root, items, results):
    by_ex = {r["example"]: r for r in results.get("results", [])}
    parts = ["<html><head><meta charset='utf-8'><title>YSDA benchmark report</title>",
             "<style>body{font-family:-apple-system,Helvetica,sans-serif;margin:24px;max-width:1800px}"
             "img{max-width:100%;border:1px solid #ddd} table{border-collapse:collapse;margin:8px 0}"
             "td,th{border:1px solid #ccc;padding:3px 8px;font-size:13px} code{background:#f4f4f4}</style></head><body>",
             f"<h1>YSDA benchmark report</h1><p>{time.strftime('%Y-%m-%d %H:%M')}. Metrics: latest "
             "<code>outputs/bench/latest.md</code> (held-out views; exact renders). Models open in ComfyUI via the "
             "<b>Load 3D</b> node (<code>3d/ysda_bench/&lt;example&gt;/&lt;method&gt;.glb</code>; press R to refresh the list) "
             "or in any GLB viewer.</p>"]
    for it in items:
        ex = it["example"]
        parts.append(f"<h2 id='{ex}'>{html.escape(ex)} ({it['kind']})</h2>")
        parts.append(f"<p>Photos: <code>{html.escape(it['photos'])}</code></p>")
        r = by_ex.get(ex)
        if r and r.get("methods"):
            keys = ["iou", "coverage", "psnr_full", "psnr_cov", "iou_low", "psnr_full_low", "chamfer_pct", "fscore_2pct"]
            keys = [k for k in keys if any(k in v for v in r["methods"].values())]
            parts.append("<table><tr><th>method</th>" + "".join(f"<th>{k}</th>" for k in keys) + "<th>model</th></tr>")
            for m, v in r["methods"].items():
                cells = "".join(f"<td>{v[k]:.3f}</td>" if k in v else "<td>–</td>" for k in keys)
                link = f"<a href='file://{it['glbs'][m]}'>{m}.glb</a>" if m in it["glbs"] else v.get("skipped", "")
                parts.append(f"<tr><td>{SHORT.get(m, m)}</td>{cells}<td>{link}</td></tr>")
            if "gt_mesh" in it["glbs"]:
                parts.append(f"<tr><td>true mesh</td>{'<td></td>' * len(keys)}<td><a href='file://{it['glbs']['gt_mesh']}'>gt_mesh.glb</a></td></tr>")
            parts.append("</table>")
        for name, title in (("inputs.jpg", "Input photos (red = held out, blue = given to the generator)"),
                            ("heldout.jpg", "Held-out photos vs each method (exact renders, PSNR inside the mask)"),
                            ("models.jpg", "Models: front, right, back, top")):
            if os.path.exists(os.path.join(out_root, ex, name)):
                parts.append(f"<h3>{title}</h3><img src='{ex}/{name}'>")
    parts.append("</body></html>")
    open(os.path.join(out_root, "index.html"), "w").write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    sys.argv = sys.argv[:1]
    sys.path[:0] = [COMFY, REPO]
    from ysda3d import bench, bench_eval
    exs = bench.load_examples([n for n in args.examples.split(",") if n] or None)
    exs = [e for e in exs if os.path.exists(os.path.join(bench.ROOT, "work", e["name"], "split.json"))]
    out_root = os.path.join(bench.ROOT, "report")
    os.makedirs(out_root, exist_ok=True)
    results_path = os.path.join(REPO, "outputs", "bench")
    runs = sorted(d for d in os.listdir(results_path) if os.path.isfile(os.path.join(results_path, d, "results.json")))
    results = json.load(open(os.path.join(results_path, runs[-1], "results.json"))) if runs else {}
    threads = max(1, (os.cpu_count() or 8) // max(1, args.workers))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init, initargs=(threads,)) as pool:
        futs = [pool.submit(_work, ex, list(bench_eval.METHODS), out_root) for ex in exs]
        items = []
        for ex, fu in zip(exs, futs):
            items.append(fu.result())
            print(f"{ex['name']}: done ({items[-1]['seconds']}s)", flush=True)
    write_html(out_root, items, results)
    print("report:", os.path.join(out_root, "index.html"))
    print("models:", GLB_ROOT)


if __name__ == "__main__":
    main()
