"""Score the approaches on every prepared example, and average across examples.

  ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python -I scripts/bench_run.py [--examples a,b] [--methods a,b]
                                                                              [--workers 3] [--quick]

Prepare the examples first (scripts/bench_prepare.py). Writes outputs/bench/<timestamp>/results.json and
summary.md, and copies the summary to outputs/bench/latest.md. Averages use only the examples on which
every compared method ran (e.g. a generation still in the queue drops that example), so rows stay comparable.
Expensive stages are cached ($YSDA_BENCH_ROOT/cache), so repeating a run takes a fraction of the first.
  --quick   one example per kind (toy, co3d_cup, gso_sofa)
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time

COMFY = os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUICK = ["toy", "co3d_cup", "gso_sofa"]
COLUMNS = {   # group -> [(metric, header, higher is better)]
    "all": [("iou", "IoU", 1), ("psnr_full", "PSNR full", 1), ("psnr_cov", "PSNR cov", 1)],
    "synthetic": [("iou_low", "IoU low", 1), ("psnr_full_low", "PSNR low", 1), ("chamfer_pct", "Chamfer %", 0),
                  ("fscore_1pct", "F@1%", 1), ("fscore_2pct", "F@2%", 1), ("precision_2pct", "Prec@2%", 1),
                  ("recall_2pct", "Rec@2%", 1)],
}


def _init(threads):
    sys.argv = sys.argv[:1]
    sys.path[:0] = [COMFY, REPO]
    import torch
    torch.set_num_threads(threads)


def _work(ex, methods):
    from ysda3d import bench_eval
    t0 = time.time()
    try:
        out = bench_eval.evaluate_example(ex, methods)
    except FileNotFoundError as e:
        out = {"example": ex["name"], "kind": ex["kind"], "methods": {}, "error": f"not prepared: {e}"}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarise(results, methods):
    import numpy as np
    lines = []
    ok = [r for r in results if r.get("methods")]
    for group, kinds in (("all examples", ("real", "synthetic")), ("real photos", ("real",)), ("synthetic (with ground truth)", ("synthetic",))):
        rs = [r for r in ok if r["kind"] in kinds]
        common = [r for r in rs if all("skipped" not in r["methods"].get(m, {"skipped": 1}) for m in methods
                                       if not (m == "oracle_gt_mesh" and r["kind"] != "synthetic"))]
        if not common:
            continue
        cols = COLUMNS["all"] + (COLUMNS["synthetic"] if kinds == ("synthetic",) else [])
        ms = [m for m in methods if not (m == "oracle_gt_mesh" and kinds != ("synthetic",))]
        lines += [f"### {group}: {len(common)} examples ({', '.join(r['example'] for r in common)})", "",
                  "| method | " + " | ".join(h for _, h, _ in cols) + " |", "|---|" + "---|" * len(cols)]
        best = {}
        for k, _, hib in cols:
            vals = {m: np.mean([r["methods"][m][k] for r in common]) for m in ms if all(k in r["methods"][m] for r in common)}
            if vals:
                best[k] = (max if hib else min)(vals, key=vals.get)
        for m in ms:
            cells = []
            for k, _, _ in cols:
                xs = [r["methods"][m][k] for r in common if k in r["methods"][m]]
                if len(xs) < len(common):
                    cells.append("–")
                    continue
                mean, sd = float(np.mean(xs)), float(np.std(xs))
                cell = f"{mean:.3f} ± {sd:.3f}" if k in ("iou", "iou_low", "fscore_1pct", "fscore_2pct", "precision_2pct", "recall_2pct") \
                    else f"{mean:.2f} ± {sd:.2f}"
                cells.append(f"**{cell}**" if best.get(k) == m and m != "oracle_gt_mesh" else cell)
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["### per example: PSNR full / IoU (held-out views)", "",
              "| method | " + " | ".join(r["example"] for r in ok) + " |", "|---|" + "---|" * len(ok)]
    for m in methods:
        cells = []
        for r in ok:
            x = r["methods"].get(m, {})
            cells.append(f"{x['psnr_full']:.2f} / {x['iou']:.3f}" if "psnr_full" in x else x.get("skipped", "–"))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    skipped = [f"{r['example']}: {r.get('error')}" for r in results if r.get("error")]
    if skipped:
        lines += ["", "Not evaluated: " + "; ".join(skipped)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sys.argv = sys.argv[:1]
    sys.path[:0] = [COMFY, REPO]
    from ysda3d import bench, bench_eval
    names = [n for n in args.examples.split(",") if n] or (QUICK if args.quick else None)
    exs = bench.load_examples(names)
    methods = [m for m in args.methods.split(",") if m] or list(bench_eval.METHODS)
    threads = max(1, (os.cpu_count() or 8) // max(1, args.workers))
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init, initargs=(threads,)) as pool:
        futs = [pool.submit(_work, ex, methods) for ex in exs]
        results = []
        for ex, fu in zip(exs, futs):
            r = fu.result()
            results.append(r)
            print(f"[{time.time() - t0:6.0f}s] {ex['name']}: {r.get('error') or 'done'} ({r['seconds']}s)", flush=True)
    summary = summarise(results, methods)
    header = (f"# Benchmark {time.strftime('%Y-%m-%d %H:%M')} ({time.time() - t0:.0f} s)\n\n"
              + "\n".join(f"- `{m}`: {bench_eval.METHODS[m][0]}" for m in methods) + "\n\n"
              "Mean ± std across examples of per-example means over held-out views. Bold = best (excluding the oracle).\n\n")
    out = os.path.join(REPO, "outputs", "bench", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    json.dump({"methods": methods, "results": results}, open(os.path.join(out, "results.json"), "w"), indent=1)
    open(os.path.join(out, "summary.md"), "w").write(header + summary + "\n")
    shutil.copy(os.path.join(out, "summary.md"), os.path.join(REPO, "outputs", "bench", "latest.md"))
    print(header + summary)


if __name__ == "__main__":
    main()
