"""Synthetic handheld captures of textured meshes, for the benchmark (journal 0016).

A scanned object (e.g. Google Scanned Objects: mesh + UVs + texture, real size) stands on a textured floor
inside a textured backdrop cylinder. Both are feature-rich, so COLMAP registers the views as it does in a
real room. Lighting: one directional light with shadows (shadow map) plus ambient. Cameras orbit like a
person walking around a table:
  - training: rings at ~25° and ~50° elevation, two near-top views;
  - held out: views between the rings ("interp"), and two low ~10° views ("low") that see more of the
    lower sides than any training view does.
Rendering is a perspective z-buffer rasteriser in torch (CPU), 2x supersampled, perspective-correct UVs.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ------------------------------------------------------------------------------------------- textures
def noise_texture(h: int, w: int, seed: int, palette, poster_count: int = 0, speckles: int = 0) -> np.ndarray:
    """Realistic but feature-rich surface: multi-octave noise on a palette, fine grain, terrazzo-like
    speckles and optional 'posters' (rectangles). Corners and blobs at many scales are what SfM matches."""
    from scipy import ndimage
    g = torch.Generator().manual_seed(seed)
    field = torch.zeros(1, 1, h, w)
    amp, tot = 1.0, 0.0
    for cells in (4, 8, 16, 32, 64, 128, 256, 512):
        grid = torch.rand(1, 1, max(2, cells * h // max(h, w)), max(2, cells * w // max(h, w)), generator=g)
        field += amp * F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
        tot += amp
        amp *= 0.85
    field = field[0, 0] / tot
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    pal = torch.tensor(palette, dtype=torch.float32) / 255.0                  # (P,3)
    x = field * (len(pal) - 1)
    i0 = x.floor().long().clamp(max=len(pal) - 2)
    t = (x - i0)[..., None]
    img = (pal[i0] * (1 - t) + pal[i0 + 1] * t).numpy()
    rng = np.random.default_rng(seed + 1)
    img *= (0.94 + 0.12 * rng.random((h, w, 1))).astype(np.float32)          # fine grain
    for size, count in ((3, speckles), (7, speckles // 6)):                    # terrazzo speckles
        if count <= 0:
            continue
        imp = np.zeros((h, w, 3), np.float32)
        ys, xs = rng.integers(0, h, count), rng.integers(0, w, count)
        imp[ys, xs] = pal.numpy()[rng.integers(0, len(pal), count)] * rng.uniform(0.5, 1.4, (count, 1))
        fp = np.hypot(*np.mgrid[-size:size + 1, -size:size + 1]) <= size
        spread = np.stack([ndimage.grey_dilation(imp[..., c], footprint=fp) for c in range(3)], -1)
        hit = ndimage.binary_dilation(imp.sum(-1) > 0, structure=fp)
        img[hit] = spread[hit]
    for _ in range(poster_count):
        ph, pw = int(rng.uniform(0.08, 0.3) * h), int(rng.uniform(0.02, 0.07) * w)
        y0, x0 = int(rng.uniform(0, h - ph)), int(rng.uniform(0, w - pw))
        c = rng.uniform(0.15, 0.9, 3).astype(np.float32)
        patch = img[y0:y0 + ph, x0:x0 + pw]
        img[y0:y0 + ph, x0:x0 + pw] = 0.6 * c + 0.4 * patch[..., ::-1]
        img[y0:y0 + ph, x0:x0 + 3] = img[y0:y0 + ph, x0 + pw - 3:x0 + pw] = 0.1    # frame edges
        img[y0:y0 + 3, x0:x0 + pw] = img[y0 + ph - 3:y0 + ph, x0:x0 + pw] = 0.1
    return img.clip(0, 1).astype(np.float32)


FLOOR_PALETTES = [[(92, 60, 38), (140, 98, 62), (181, 140, 96), (120, 82, 50), (60, 42, 30)],
                  [(70, 74, 82), (120, 124, 130), (170, 168, 160), (96, 92, 88), (40, 42, 46)],
                  [(150, 120, 90), (196, 170, 132), (226, 210, 180), (170, 140, 104), (110, 84, 60)]]
WALL_PALETTES = [[(210, 206, 196), (182, 178, 170), (232, 228, 220), (160, 156, 150)],
                 [(196, 208, 216), (160, 176, 190), (220, 228, 232), (138, 150, 160)]]


# ------------------------------------------------------------------------------------------- scene
def _grid_faces(nu: int, nv: int) -> np.ndarray:
    i = (np.arange(nv)[:, None] * (nu + 1) + np.arange(nu)[None]).ravel()
    return np.concatenate([np.stack([i, i + nu + 1, i + 1], 1), np.stack([i + 1, i + nu + 1, i + nu + 2], 1)])


def floor_mesh(half: float = 1.5, n: int = 120):
    xs = np.linspace(-half, half, n + 1)
    X, Z = np.meshgrid(xs, xs)
    v = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    uv = np.stack([(X.ravel() + half) / (2 * half), (Z.ravel() + half) / (2 * half)], 1)
    return v, _grid_faces(n, n), uv


def backdrop_mesh(radius: float = 1.4, height: float = 2.0, n_seg: int = 160, n_ring: int = 30):
    th = np.linspace(0, 2 * math.pi, n_seg + 1)
    ys = np.linspace(0, height, n_ring + 1)
    T, Y = np.meshgrid(th, ys)
    v = np.stack([radius * np.cos(T.ravel()), Y.ravel(), radius * np.sin(T.ravel())], 1)
    uv = np.stack([T.ravel() / (2 * math.pi), 1 - Y.ravel() / height], 1)
    return v, _grid_faces(n_seg, n_ring), uv


def place_object(verts: np.ndarray, max_extent_m: float, yaw_deg: float, up: str = "z") -> np.ndarray:
    """Normalised mesh (max extent 1) -> metres, +Y up, resting on the floor (y = 0), centred at x = z = 0."""
    v = verts * max_extent_m
    if up == "z":                                   # GSO GLBs keep the scans' Z-up axes
        v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
    a = math.radians(yaw_deg)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    v = v @ R.T
    c = (v.max(0) + v.min(0)) / 2
    return v - np.array([c[0], v[:, 1].min(), c[2]])


# ------------------------------------------------------------------------------------------- cameras
def look_at(center: np.ndarray, target: np.ndarray, roll_deg: float = 0.0):
    """World->camera [R|t] (OpenCV axes: x right, y down, z forward) for a camera at `center`."""
    z = target - center
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    y = -up + np.dot(up, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    r = math.radians(roll_deg)
    x, y = math.cos(r) * x + math.sin(r) * y, -math.sin(r) * x + math.cos(r) * y
    R = np.stack([x, y, z])
    return R, -R @ center


def capture_cameras(target: np.ndarray, radius: float, seed: int, W: int = 1152, H: int = 864,
                    fov_x_deg: float = 65.0, fill: float = 0.75):
    """Training + held-out cameras of a handheld-style capture. Returns K and a list of (name, R, t, role)."""
    rng = np.random.default_rng(seed)
    fx = (W / 2) / math.tan(math.radians(fov_x_deg) / 2)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    dist0 = radius / (fill * (H / 2) / fx)          # bounding sphere spans ~fill of the image height
    az0 = rng.uniform(0, 360)
    plan = [("t", 25, az0 + 36 * i) for i in range(10)] + [("t", 50, az0 + 18 + 45 * i) for i in range(8)]
    plan += [("t", 75, az0 + 90 + 180 * i) for i in range(2)]
    plan += [("h_interp", 37, az0 + 60 + 120 * i) for i in range(3)] + [("h_low", 10, az0 + 30 + 180 * i) for i in range(2)]
    cams, counts = [], {}
    for role, el, az in plan:
        el = el + rng.uniform(-3, 3)
        az = math.radians(az + rng.uniform(-5, 5))
        d = dist0 * rng.uniform(0.9, 1.1)
        elr = math.radians(el)
        c = target + d * np.array([math.cos(elr) * math.sin(az), math.sin(elr), math.cos(elr) * math.cos(az)])
        R, t = look_at(c, target + rng.uniform(-0.01, 0.01, 3) * radius / 0.1, roll_deg=rng.uniform(-3, 3))
        k = counts.get(role, 0)
        counts[role] = k + 1
        cams.append((f"{role}_{k:02d}", R, t, role))
    return K, cams


# ------------------------------------------------------------------------------------------- rasteriser
def rasterize(u, v, zinv, faces, H, W, boxes=(4, 12, 32, 96, 256, 1024)):
    """Frontmost face per pixel centre and its screen barycentrics, for screen coords u, v (pixels) and
    1/depth `zinv` per vertex (larger = closer; use 1/z for perspective, -depth for orthographic)."""
    U, V, Z = u[faces], v[faces], zinv[faces]
    lo_u, hi_u = (U.min(1).values - 0.5).ceil(), (U.max(1).values - 0.5).floor()
    lo_v, hi_v = (V.min(1).values - 0.5).ceil(), (V.max(1).values - 0.5).floor()
    keep = (hi_u >= 0) & (lo_u < W) & (hi_v >= 0) & (lo_v < H) & (hi_u >= lo_u) & (hi_v >= lo_v)
    lo_u, lo_v = lo_u.clamp(min=0), lo_v.clamp(min=0)
    size = torch.maximum(hi_u.clamp(max=W - 1) - lo_u, hi_v.clamp(max=H - 1) - lo_v) + 1
    best = torch.full((H * W,), -float("inf"))
    fid = torch.full((H * W,), -1, dtype=torch.long)
    bary = torch.zeros(H * W, 3)
    prev = 0
    for box in boxes:
        sel = torch.nonzero(keep & (size > prev) & (size <= box)).squeeze(1)
        prev = box
        if len(sel) == 0:
            continue
        oy, ox = torch.meshgrid(torch.arange(box), torch.arange(box), indexing="ij")
        ox, oy = ox.reshape(-1).float(), oy.reshape(-1).float()
        chunk = max(1, 24_000_000 // (box * box))
        for k in range(0, len(sel), chunk):
            idx = sel[k:k + chunk]
            pu, pv = lo_u[idx, None] + ox[None], lo_v[idx, None] + oy[None]
            cu, cv = pu + 0.5, pv + 0.5
            u0, u1, u2 = U[idx, 0:1], U[idx, 1:2], U[idx, 2:3]
            v0, v1, v2 = V[idx, 0:1], V[idx, 1:2], V[idx, 2:3]
            den = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            ok = den.abs() > 1e-12
            den = torch.where(ok, den, torch.ones_like(den))
            l0 = ((v1 - v2) * (cu - u2) + (u2 - u1) * (cv - v2)) / den
            l1 = ((v2 - v0) * (cu - u2) + (u0 - u2) * (cv - v2)) / den
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6) & ok & (pu < W) & (pv < H)
            zz = l0 * Z[idx, 0:1] + l1 * Z[idx, 1:2] + l2 * Z[idx, 2:3]
            pix = (pv * W + pu).long()[inside]
            zv = zz[inside]
            best.scatter_reduce_(0, pix, zv, reduce="amax")
            win = zv >= best[pix]
            pw = pix[win]
            fid[pw] = idx[:, None].expand_as(inside)[inside][win]
            bary[pw] = torch.stack([l0[inside][win], l1[inside][win], l2[inside][win]], 1)
    return fid.reshape(H, W), bary.reshape(H, W, 3)


def _sample(tex: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup; tex (H,W,3), uv (N,2) with v = 0 at the image top (glTF)."""
    grid = (uv * 2 - 1)[None, None]
    return F.grid_sample(tex.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border",
                         align_corners=False)[0, :, 0].T


class Scene:
    """Object + floor + backdrop, with materials; renders photos, object masks and depth."""

    def __init__(self, obj_v, obj_f, obj_uv, obj_tex, seed: int, light_dir=(0.35, 1.0, 0.45), ambient: float = 0.42):
        rng = np.random.default_rng(seed)
        fv, ff, fuv = floor_mesh()
        bv, bf, buv = backdrop_mesh()
        nobj, nfl = len(obj_v), len(fv)
        self.v = torch.from_numpy(np.concatenate([obj_v, fv, bv]).astype(np.float32))
        self.f = torch.from_numpy(np.concatenate([obj_f, ff + nobj, bf + nobj + nfl]).astype(np.int64))
        self.uv = torch.from_numpy(np.concatenate([obj_uv, fuv, buv]).astype(np.float32))
        self.mat = torch.cat([torch.zeros(len(obj_f), dtype=torch.long), torch.ones(len(ff), dtype=torch.long),
                              torch.full((len(bf),), 2, dtype=torch.long)])
        floor_tex = noise_texture(4096, 4096, seed * 7 + 1, FLOOR_PALETTES[rng.integers(len(FLOOR_PALETTES))],
                                  speckles=120_000)
        wall_tex = noise_texture(1024, 4096, seed * 7 + 2, WALL_PALETTES[rng.integers(len(WALL_PALETTES))],
                                 poster_count=70, speckles=6_000)
        self.tex = [torch.as_tensor(obj_tex, dtype=torch.float32), torch.from_numpy(floor_tex), torch.from_numpy(wall_tex)]
        tri = self.v[self.f]
        n = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
        self.fn = F.normalize(n, dim=-1)
        l = torch.tensor(light_dir, dtype=torch.float32)
        self.l = l / l.norm()
        self.ambient = ambient
        self.n_obj_faces = len(obj_f)
        self._shadow_map()

    def _shadow_map(self, res: int = 2048):
        """Orthographic depth of the object as seen from the light (for shadows on floor and object)."""
        l = self.l
        a = torch.tensor([1.0, 0.0, 0.0]) if abs(float(l[0])) < 0.9 else torch.tensor([0.0, 0.0, 1.0])
        x = F.normalize(torch.linalg.cross(a, l, dim=0), dim=0)
        y = torch.linalg.cross(l, x, dim=0)
        ov = self.v[: int(self.f[: self.n_obj_faces].max()) + 1]
        px, py = ov @ x, ov @ y
        self._sm_basis = (x, y)
        self._sm_lo = torch.stack([px.min(), py.min()]) - 0.02
        self._sm_scale = (res - 1) / float(max(px.max() - px.min(), py.max() - py.min()) + 0.04)
        u = (px - self._sm_lo[0]) * self._sm_scale
        v = (py - self._sm_lo[1]) * self._sm_scale
        fid, bary = rasterize(u, v, ov @ l, self.f[: self.n_obj_faces], res, res)
        h = (bary * (ov @ l)[self.f[: self.n_obj_faces]][fid.clamp(min=0)]).sum(-1)
        self._sm = torch.where(fid >= 0, h, torch.full_like(h, -float("inf")))   # height towards the light
        self._sm_res = res

    def _lit(self, P: torch.Tensor) -> torch.Tensor:
        """1 where P (N,3) sees the light, 0 where the object blocks it."""
        x, y = self._sm_basis
        u = ((P @ x - self._sm_lo[0]) * self._sm_scale).round().long()
        v = ((P @ y - self._sm_lo[1]) * self._sm_scale).round().long()
        inside = (u >= 0) & (u < self._sm_res) & (v >= 0) & (v < self._sm_res)
        occl = torch.full((len(P),), -float("inf"))
        occl[inside] = self._sm[v[inside], u[inside]]
        bias = 0.004
        return ((P @ self.l) >= occl - bias).float()

    def render(self, K: np.ndarray, R: np.ndarray, t: np.ndarray, W: int, H: int, ss: int = 2, near: float = 0.02):
        """Photo (H,W,3 in [0,1]), object mask (H,W in [0,1]) and object depth (H,W, 0 = background)."""
        Kt = torch.as_tensor(K, dtype=torch.float32) * torch.tensor([[ss], [ss], [1.0]])
        Rt, tt = torch.as_tensor(R, dtype=torch.float32), torch.as_tensor(t, dtype=torch.float32)
        Xc = self.v @ Rt.T + tt
        z = Xc[:, 2]
        good = (z[self.f] > near).all(1)
        faces = self.f[good]
        zs = z.clamp(min=near)
        u = Kt[0, 0] * Xc[:, 0] / zs + Kt[0, 2]
        v = Kt[1, 1] * Xc[:, 1] / zs + Kt[1, 2]
        Hs, Ws = H * ss, W * ss
        fid, bary = rasterize(u, v, 1.0 / zs, faces, Hs, Ws)
        cov = fid >= 0
        gfi = torch.nonzero(good).squeeze(1)[fid[cov]]               # global face index per covered pixel
        b = bary[cov]
        fz = zs[self.f[gfi]]                                         # perspective-correct weights
        w = b / fz
        w = w / w.sum(1, keepdim=True)
        uvp = (self.uv[self.f[gfi]] * w[..., None]).sum(1)
        P = (self.v[self.f[gfi]] * w[..., None]).sum(1)
        mat = self.mat[gfi]
        col = torch.zeros(len(gfi), 3)
        for m in range(3):
            s = mat == m
            if s.any():
                col[s] = _sample(self.tex[m], uvp[s])
        n = self.fn[gfi]
        view = F.normalize(P - torch.as_tensor(-R.T @ t, dtype=torch.float32), dim=-1)
        n = torch.where(((n * view).sum(-1, keepdim=True) > 0), -n, n)     # face the camera
        lit = self._lit(P)
        shade = self.ambient + (1 - self.ambient) * (n @ self.l).clamp(min=0) * lit
        lin = col.clamp(0, 1) ** 2.2 * shade[:, None]
        img = torch.full((Hs * Ws, 3), 0.75)
        img[cov.reshape(-1)] = lin.clamp(0, 1) ** (1 / 2.2)
        objm = torch.zeros(Hs * Ws)
        objm[cov.reshape(-1)] = (mat == 0).float()
        dep = torch.zeros(Hs * Ws)
        dep[cov.reshape(-1)] = torch.where(mat == 0, (fz * w).sum(1), torch.zeros_like(fz[:, 0]))   # camera depth
        img = img.reshape(Hs, Ws, 3).permute(2, 0, 1)[None]
        img = F.avg_pool2d(img, ss)[0].permute(1, 2, 0)
        objm = F.avg_pool2d(objm.reshape(1, 1, Hs, Ws), ss)[0, 0]
        dep = F.max_pool2d(dep.reshape(1, 1, Hs, Ws), ss)[0, 0]
        return img.numpy(), objm.numpy(), dep.numpy()
