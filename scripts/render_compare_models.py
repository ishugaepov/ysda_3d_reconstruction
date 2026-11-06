"""Render several 3D models (GLB/OBJ, textured or vertex-coloured) from the same orbit of viewpoints.

Used to compare our output with reference apps (KIRI, Polycam). Each model is centred and scaled to
unit size; "up" is the file's +Y (glTF convention). An optional per-model rotation fixes other axes.

  ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python scripts/render_compare_models.py out.jpg \
      ours=path/a.glb kiri=path/3DModel.obj@path/3DModel.jpg poly=path/b.glb
"""
import math
import os
import sys

COMFY = os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path, specs = sys.argv[1], sys.argv[2:]
sys.argv = sys.argv[:1]
sys.path[:0] = [COMFY, REPO]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from comfy_api.latest._util.geometry_types import File3D  # noqa: E402
from comfy_extras.nodes_mesh_io import Get3DComponents  # noqa: E402
from ysda3d.render import render_points  # noqa: E402

torch.set_grad_enabled(False)


def load(spec):
    path, _, tex_path = spec.partition("@")
    mesh = Get3DComponents.execute(File3D(path)).result[0]
    V = mesh.vertices[0].float()
    Fc = mesh.faces[0].long()
    uv = mesh.uvs[0].float() if mesh.uvs is not None else None
    tex = mesh.texture[0].float() if mesh.texture is not None else None
    if tex_path:
        tex = torch.from_numpy(np.asarray(Image.open(tex_path).convert("RGB"), np.float32) / 255)
    vc = mesh.vertex_colors[0].float()[:, :3] if mesh.vertex_colors is not None else None
    return V, Fc, uv, tex, vc


def surface_samples(V, Fc, uv, tex, vc, n=1_500_000):
    """Area-weighted random points on the triangles, with interpolated colour."""
    tri = V[Fc]
    area = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    fi = torch.multinomial(area / area.sum(), n, replacement=True)
    r1, r2 = torch.rand(n).sqrt(), torch.rand(n)
    w = torch.stack([1 - r1, r1 * (1 - r2), r1 * r2], -1)
    P = (tri[fi] * w[..., None]).sum(1)
    if tex is not None and uv is not None:
        st = (uv[Fc[fi]] * w[..., None]).sum(1)
        H, W = tex.shape[:2]
        x = (st[:, 0] % 1.0 * (W - 1)).long()
        y = (st[:, 1] % 1.0 * (H - 1)).long()   # ComfyUI MESH uvs use glTF convention (origin top-left)
        C = tex[y, x, :3]
    elif vc is not None:
        C = (vc[Fc[fi]] * w[..., None]).sum(1)
    else:
        C = torch.full((n, 3), 0.7)
    return P, C


def look_at(eye, target=np.zeros(3), up=np.array([0, 1.0, 0])):
    """World→camera (OpenCV) [R|t] for a camera at eye looking at target, with world +Y up."""
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)   # camera y (down)
    R = np.stack([r, d, f])
    return torch.tensor(np.concatenate([R, -(R @ eye)[:, None]], 1), dtype=torch.float32)


size = 360
K = torch.tensor([[size * 1.4, 0, size / 2], [0, size * 1.4, size / 2], [0, 0, 1.0]])
views = [(az, 25) for az in range(0, 360, 60)]
rows = []
for spec in specs:
    name, _, rest = spec.partition("=")
    rest, _, zoom = rest.partition("#")       # optional "#zoom" to crop away a floor patch
    V, Fc, uv, tex, vc = load(rest)
    lo, hi = V.min(0).values, V.max(0).values
    V = (V - (lo + hi) / 2) / (hi - lo).max() * float(zoom or 1)
    P, C = surface_samples(V, Fc, uv, tex, vc)
    tiles = []
    for az, el in views:
        a, e = math.radians(az), math.radians(el)
        eye = 2.2 * np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
        tiles.append(render_points(P, C, K, look_at(eye), (size, size), radius=0))
    row = Image.fromarray((np.concatenate(tiles, 1) * 255).clip(0, 255).astype(np.uint8))
    ImageDraw.Draw(row).text((6, 6), f"{name}: {len(V)} verts, {len(Fc)} faces", fill="black")
    rows.append(row)
    print(name, len(V), "verts", len(Fc), "faces", "textured" if tex is not None else "vertex colours")
sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.height
sheet.save(out_path, quality=88)
print("saved", out_path)
