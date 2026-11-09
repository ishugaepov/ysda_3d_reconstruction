"""Decimate a vertex-coloured GLB to a target face count, carrying colours by nearest original vertex.

  ~/src/uniforge3d-venv/bin/python scripts/decimate_glb.py in.glb out.glb 300000
"""
import sys

import fast_simplification
import numpy as np
import trimesh
from scipy.spatial import cKDTree

src, dst, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = trimesh.load(src, force="mesh", process=False)
n0 = len(m.vertices)
m.merge_vertices()            # exported meshes can duplicate vertices per face, which blocks edge collapses
print(f"merged vertices: {n0} -> {len(m.vertices)}")
v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)
cols = np.asarray(m.visual.vertex_colors) if hasattr(m.visual, "vertex_colors") else None
v2, f2 = v, f
for agg in (7, 10, 14):        # dual-grid meshes have non-manifold bits; raise aggressiveness until on target
    if len(f2) <= target * 1.05:
        break
    v2, f2 = fast_simplification.simplify(v2, f2, target_reduction=1.0 - target / len(f2), agg=agg)
    print(f"  agg {agg}: {len(f2)} faces")
out = trimesh.Trimesh(v2, f2, process=False)
if cols is not None:
    _, idx = cKDTree(v).query(v2)
    out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=cols[idx])
out.export(dst)
print(f"{src}: {len(f)} -> {len(f2)} faces -> {dst}")
