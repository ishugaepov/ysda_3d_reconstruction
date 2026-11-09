"""Watertight remeshing of messy generated meshes (replaces CuMesh's narrow-band remesh on Mac).

UniForge3D's dual-grid output (via our Mac port, which skips CuMesh remeshing) is non-manifold:
thousands of tiny components, many edges shared by >2 faces, small holes (journal 0012). This turns it
into a closed manifold surface: voxelise the surface, dilate to bridge small gaps, flood-fill the
outside, erode back, extract an iso-surface. Colours are carried over by nearest original vertex.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree

from . import recon


def remesh_watertight(verts: torch.Tensor, faces: torch.Tensor, surface_fn, colors: torch.Tensor | None = None,
                      res: int = 256, close_iters: int = 2, solid_close_iters: int = 3,
                      samples_per_voxel: float = 8.0, log=print):
    v = verts.float()
    lo, hi = v.min(0).values, v.max(0).values
    pad = 3 + close_iters
    voxel = float((hi - lo).max()) / (res - 2 * pad)
    origin = lo - pad * voxel
    dims = [int(np.ceil(float(e) / voxel)) + 2 * pad for e in (hi - lo)]
    # Surface voxelisation by dense sampling (enough samples that every surface voxel is hit).
    area = float(torch.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]], dim=-1).norm(dim=-1).sum() / 2)
    n = int(min(30_000_000, max(1_000_000, samples_per_voxel * area / voxel ** 2)))
    P, _ = recon.surface_samples(v, faces.long(), n)
    ijk = ((P - origin) / voxel).floor().long()
    ijk = torch.minimum(ijk.clamp(min=0), torch.tensor(dims) - 1).numpy()
    surf = np.zeros(dims, bool)
    surf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    struct = ndimage.generate_binary_structure(3, 1)
    shell = ndimage.binary_dilation(surf, struct, iterations=close_iters)
    lab, _ = ndimage.label(~shell, struct)
    outside_labels = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                               lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    outside = np.isin(lab, outside_labels[outside_labels > 0])
    solid = ~outside
    if close_iters > 0:
        solid = ndimage.binary_erosion(solid, struct, iterations=close_iters, border_value=0)
    if solid_close_iters > 0:
        # Seal thin slots that connect to the outside (e.g. between the toy's face plate and body):
        # flood fill treats them as outside, and both slot walls would become surface (journal 0012).
        solid = ndimage.binary_closing(np.pad(solid, solid_close_iters), struct, iterations=solid_close_iters)[
            solid_close_iters:-solid_close_iters, solid_close_iters:-solid_close_iters, solid_close_iters:-solid_close_iters]
    log(f"   remesh: grid {dims}, {n/1e6:.1f}M samples, solid voxels {int(solid.sum())}")
    vol = recon.smooth_volume(torch.from_numpy(solid.astype(np.float32)), 1)
    nv, nf = surface_fn(vol, 0.5, voxel, (origin + 0.5 * voxel).tolist())
    nv, nf = recon.largest_component(np.asarray(nv, np.float32), np.asarray(nf, np.int64))
    nv_t, nf_t = torch.from_numpy(nv), torch.from_numpy(nf)
    new_colors = None
    if colors is not None:
        new_colors = transfer_colors(v, faces, colors, nv_t, nf_t, max_dist=3 * voxel)
    return nv_t, nf_t, new_colors


def transfer_colors(src_v, src_f, src_c, dst_v, dst_f, k: int = 16, max_dist: float | None = None):
    """Per-vertex colour transfer that respects orientation.

    Plain nearest-vertex transfer picks up colours from inner shells of a messy mesh (e.g. face
    colours on the back of the toy, journal 0012). Among the k nearest source vertices, prefer those
    whose normal agrees with the destination's outward normal; fall back to the nearest one.
    """
    sn = recon.vertex_normals(src_v.float(), src_f.long())
    dn = recon.vertex_normals(dst_v.float(), dst_f.long())
    if ((dst_v - dst_v.mean(0)) * dn).sum() < 0:
        dn = -dn
    dist, idx = cKDTree(src_v.numpy()).query(dst_v.numpy(), k=k)
    idx_t = torch.from_numpy(idx)
    agree = (sn[idx_t] * dn[:, None, :]).sum(-1)                     # (N,k) cos between normals
    d = torch.from_numpy(dist).float()
    score = torch.where(agree > 0.3, -d, torch.full_like(d, -1e9))  # nearest among agreeing candidates
    if max_dist is not None:
        score = torch.where(d <= max_dist, score, torch.full_like(score, -1e9))
    best = score.argmax(1)
    ok = score.max(1).values > -1e8
    chosen = torch.where(ok, idx_t[torch.arange(len(idx_t)), best], idx_t[:, 0])
    return src_c[chosen]


def edge_stats(faces: torch.Tensor) -> dict:
    """Open (boundary) / non-manifold edges and winding consistency: the numbers behind journal 0013."""
    f = faces.long().cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces, np.int64)
    he = np.stack([f, np.roll(f, -1, 1)], -1).reshape(-1, 2)
    und, inv, cnt = np.unique(np.sort(he, 1), axis=0, return_inverse=True, return_counts=True)
    fwd = np.bincount(inv.reshape(-1), weights=(he[:, 0] < he[:, 1]).astype(np.float64), minlength=len(und))
    manifold = cnt == 2
    return {"faces": int(len(f)), "edges": int(len(und)), "open_edges": int((cnt == 1).sum()),
            "non_manifold_edges": int((cnt > 2).sum()),
            "inconsistent_winding_frac": round(float((manifold & (fwd != 1)).sum() / max(manifold.sum(), 1)), 4)}


def signed_volume(verts: torch.Tensor, faces: torch.Tensor) -> float:
    """Volume enclosed by a closed mesh: positive if faces are wound outward."""
    t = verts.double()[faces.long()]
    return float(torch.linalg.det(t).sum() / 6)


def cull_hidden_faces(verts: torch.Tensor, faces: torch.Tensor, n_dirs: int = 96, res: int = 768,
                      samples_per_face: int = 3, tol_frac: float = 0.002, extra_cams=None, log=print):
    """Keep only faces that are frontmost from at least one viewpoint (removes inner shells/buried fins).

    Viewpoints: a Fibonacci sphere of n_dirs orthographic directions around the mesh, plus optional real
    cameras (list of (K, E, (H, W))). A face counts as visible if any of its sample points matches the
    z-buffer at its pixel within tol_frac of the mesh size.
    """
    v, f = verts.float(), faces.long()
    visible = face_visibility(v, f, n_dirs, res, samples_per_face, tol_frac, extra_cams)
    log(f"   cull: kept {int(visible.sum())} / {len(f)} faces")
    used = torch.unique(f[visible])
    remap = torch.full((len(v),), -1, dtype=torch.long)
    remap[used] = torch.arange(len(used))
    return v[used], remap[f[visible]], used


def face_visibility(v: torch.Tensor, f: torch.Tensor, n_dirs: int = 96, res: int = 768, samples_per_face: int = 3,
                    tol_frac: float = 0.002, extra_cams=None) -> torch.Tensor:
    """Per-face bool: frontmost (within tol_frac of the mesh size) from at least one viewpoint."""
    c = (v.max(0).values + v.min(0).values) / 2
    ext = float((v.max(0).values - v.min(0).values).norm())
    g = torch.Generator().manual_seed(0)
    w = torch.rand(len(f), samples_per_face, 3, generator=g)
    w = w / w.sum(-1, keepdim=True)
    P = (v[f][:, None, :, :] * w[..., None]).sum(2).reshape(-1, 3)        # (F*s, 3)
    P = torch.cat([P, v[f].mean(1)])                                       # + centroids
    fid = torch.cat([torch.arange(len(f)).repeat_interleave(samples_per_face), torch.arange(len(f))])
    visible = torch.zeros(len(f), dtype=torch.bool)
    tol = tol_frac * ext
    i = torch.arange(n_dirs, dtype=torch.float32) + 0.5
    phi = torch.arccos(1 - 2 * i / n_dirs)
    theta = math.pi * (1 + 5 ** 0.5) * i
    dirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], -1)

    def mark(uv, z, H, W):
        u, vv = uv[:, 0].round().long(), uv[:, 1].round().long()
        ok = (z > 0) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        pix = vv[ok] * W + u[ok]
        zb = torch.full((H * W,), float("inf"))
        zb.scatter_reduce_(0, pix, z[ok], reduce="amin")
        hit = (z[ok] - zb[pix]) < tol
        visible[fid[ok][hit]] = True

    for d in dirs:                                   # orthographic views
        a = torch.tensor([0.0, 1.0, 0.0]) if abs(float(d[1])) < 0.9 else torch.tensor([1.0, 0.0, 0.0])
        x = F.normalize(torch.cross(a, d, dim=0), dim=0)
        y = torch.cross(d, x, dim=0)
        q = P - c
        uv = torch.stack([(q @ x) / ext * res + res / 2, (q @ y) / ext * res + res / 2], -1)
        mark(uv, q @ d + ext, res, res)
    for K, E, hw in (extra_cams or []):              # real cameras (pinhole)
        uv, z = recon.project(P, K, E)
        mark(uv, z, *hw)
    return visible


def remesh_narrow_band(verts: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor | None = None,
                       res: int = 768, band: float = 1.0, seal: float = 1.0, close_bottom: bool = True,
                       samples_per_voxel: float = 6.0, max_samples: int = 30_000_000,
                       min_component_area: float = 1e-5, log=print):
    """Clean, closed, outward-wound surface around a messy one: a CPU stand-in for CuMesh's hole filling +
    remesh_narrow_band_dc, which UniForge3D runs after mesh extraction and our Mac port lacked (journal 0013).

    The surface is the outer boundary of the band {x : dist(x, input) < band voxels} on a grid with `res`
    cells along the longest side (sparse distances, dense masks, surface nets):
      - holes and cracks narrower than ~2·band voxels close, overlapping patches merge into one sheet;
      - "outside" is flood-filled from the grid border through points at least band + seal voxels from
        the input and grown back by seal, so gaps up to ~2·(band + seal) voxels wide are capped rather
        than leaving tunnels to the interior, and no inner shell is produced;
      - no morphological closing of the solid (unlike remesh_watertight): layers further apart than
        ~2·(band + seal) voxels, like the toy's face plate and the body, stay separate;
      - close_bottom: points from which every ray that doesn't point down (8 horizontal, 8 at 45° up,
        straight up) hits the surface count as interior. That caps openings that only face down
        (UniForge3D leaves the never-photographed underside open) at about their rim; anything that
        opens sideways or upwards, like the pockets between the toy's fins and body, stays open. +Y is up.
    Specks smaller than min_component_area of the total area are dropped. Returns (verts, faces, colours).
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    v, f = verts.float(), faces.long()
    lo, hi = v.min(0).values, v.max(0).values
    voxel = float((hi - lo).max()) / res
    eps = band * voxel
    pad = int(math.ceil(band + seal)) + 3
    origin = (lo.double() - pad * voxel).numpy()
    dims = [int(math.ceil(float(e) / voxel)) + 2 * pad + 1 for e in (hi - lo)]
    area = float(torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1).norm(dim=-1).sum()) / 2
    n = int(min(max_samples, max(1_000_000, samples_per_voxel * area / voxel ** 2)))
    P, C = recon.surface_samples(v, f, n, colors.float()[:, :3] if colors is not None else None)
    Pn = P.double().numpy()
    del P

    # Unsigned distance on the grid vertices near the surface; farther ones only need a lower bound.
    r = int(math.ceil(band + seal)) + 1
    near = np.zeros(dims, bool)
    ijk = np.rint((Pn - origin) / voxel).astype(np.int64)
    near[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    del ijk
    near = ndimage.binary_dilation(near, ndimage.generate_binary_structure(3, 3), iterations=r)
    bi = np.nonzero(near)
    del near
    tree = cKDTree(Pn)
    phi = np.full(dims, np.float32((r - 0.5) * voxel - eps))        # dist - eps; > 0 is off the band
    phi[bi] = tree.query(origin + np.stack(bi, 1) * voxel, workers=-1)[0] - eps
    log(f"   remesh: grid {dims} (voxel {voxel:.5f}), {n / 1e6:.1f}M samples, {len(bi[0]) / 1e6:.1f}M distances")
    del bi

    # Outside = reachable from the grid border through points >= seal voxels off the band, grown back.
    free = phi >= seal * voxel
    if close_bottom:
        free &= ~_enclosed_from_above(~free)
    lab, _ = ndimage.label(free)
    del free
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    out = np.isin(lab, border[border > 0])
    del lab
    if seal > 0:
        out = ndimage.binary_dilation(out, ndimage.generate_binary_structure(3, 3), iterations=int(math.ceil(seal)) + 1)
    out &= phi >= 0
    psi = np.where(out | (phi < 0), phi, np.float32(-0.5 * voxel))  # sign of psi == outside; caps sit half-way

    # One quad per grid edge between outside and inside, joining the 4 cells (named by their min corner)
    # around the edge and wound so its normal points to the outside end.
    quads = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        e, eb, ec = (np.eye(3, dtype=np.int64)[k] for k in (a, b, c))
        ring = np.stack([0 * e, -eb, -eb - ec, -ec])    # counter-clockwise about +a
        lo_sl = [slice(None)] * 3
        hi_sl = [slice(None)] * 3
        lo_sl[a], hi_sl[a] = slice(0, -1), slice(1, None)
        o0, o1 = out[tuple(lo_sl)], out[tuple(hi_sl)]
        for outward_plus in (True, False):
            start = np.stack(np.nonzero(~o0 & o1 if outward_plus else o0 & ~o1), 1)
            cells = start[:, None, :] + ring[None]
            quads.append(cells if outward_plus else cells[:, ::-1])
    del out
    quads = np.concatenate(quads)
    D1, D2 = dims[1], dims[2]
    cell_keys, qidx = np.unique(((quads[..., 0] * D1 + quads[..., 1]) * D2 + quads[..., 2]).reshape(-1),
                                return_inverse=True)
    qidx = qidx.reshape(-1, 4)
    del quads
    cells = np.stack([cell_keys // (D1 * D2), (cell_keys // D2) % D1, cell_keys % D2], 1)

    # Surface-nets vertex per cell: mean of the zero crossings of psi on the cell's 12 edges.
    corner = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], np.int64)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.abs(corner[i] - corner[j]).sum() == 1]
    nv = np.empty((len(cells), 3))
    step = 2_000_000
    for s0 in range(0, len(cells), step):
        cc = cells[s0:s0 + step]
        q = cc[:, None, :] + corner[None]
        ph = psi[q[..., 0], q[..., 1], q[..., 2]]
        acc = np.zeros((len(cc), 3))
        cnt = np.zeros(len(cc))
        for i, j in edges:
            ch = (ph[:, i] < 0) != (ph[:, j] < 0)
            t = np.where(ch, ph[:, i] / np.where(ch, ph[:, i] - ph[:, j], 1), 0)
            acc += ch[:, None] * (corner[i] + t[:, None] * (corner[j] - corner[i]))
            cnt += ch
        nv[s0:s0 + step] = origin + (cc + acc / np.maximum(cnt, 1)[:, None]) * voxel
    del psi, phi
    # split each quad along its shorter diagonal (both halves keep the quad's winding)
    d02 = np.linalg.norm(nv[qidx[:, 0]] - nv[qidx[:, 2]], axis=1)
    d13 = np.linalg.norm(nv[qidx[:, 1]] - nv[qidx[:, 3]], axis=1)
    use02 = (d02 <= d13)[:, None]
    nf = np.concatenate([np.where(use02, qidx[:, [0, 1, 2]], qidx[:, [0, 1, 3]]),
                         np.where(use02, qidx[:, [0, 2, 3]], qidx[:, [1, 2, 3]])])

    # Drop specks (small connected components).
    fe = np.sort(np.stack([nf, np.roll(nf, -1, 1)], -1).reshape(-1, 2), 1)
    ek = fe[:, 0] * len(nv) + fe[:, 1]
    order = np.argsort(ek, kind="stable")
    same = ek[order][1:] == ek[order][:-1]
    fid = np.repeat(np.arange(len(nf)), 3)[order]
    adj = coo_matrix((np.ones(int(same.sum())), (fid[1:][same], fid[:-1][same])), shape=(len(nf), len(nf)))
    ncomp, comp = connected_components(adj, directed=False)
    fa = np.linalg.norm(np.cross(nv[nf[:, 1]] - nv[nf[:, 0]], nv[nf[:, 2]] - nv[nf[:, 0]]), axis=1) / 2
    comp_area = np.bincount(comp, weights=fa, minlength=ncomp)
    keep_comp = comp_area >= min_component_area * comp_area.sum()
    keep = keep_comp[comp]
    used, remap = np.unique(nf[keep], return_inverse=True)
    out_v = torch.from_numpy(nv[used].astype(np.float32))
    out_f = torch.from_numpy(remap.reshape(-1, 3).astype(np.int64))
    out_c = None
    if C is not None:
        # nearest sample within a few cells (far queries, e.g. the bottom cap, are slow and meaningless);
        # the rest inherit colour by diffusion over the mesh from coloured neighbours
        d, nn = tree.query(nv[used], workers=-1, distance_upper_bound=(band + seal + 2) * voxel)
        has = np.isfinite(d)
        col = np.zeros((len(used), 3), np.float32)
        col[has] = C.numpy()[nn[has]]
        if not has.all():
            col = _diffuse_colours(remap.reshape(-1, 3), col, has)
        out_c = torch.from_numpy(col)
        log(f"   remesh: {int((~has).sum())} vertices away from the input (caps) got diffused colours")
    log(f"   remesh: {len(nf)} faces in {ncomp} components -> kept {int(keep_comp.sum())}, {len(out_f)} faces")
    return out_v, out_f, out_c


def _enclosed_from_above(wall: np.ndarray) -> np.ndarray:
    """Grid points from which all 17 non-downward rays (steps in {-1,0,1} x {0,1} x {-1,0,1}) hit `wall`.

    Computed on a 2x coarser grid (wall max-pooled, so thin walls still block), by log-doubling shifts.
    """
    dims = wall.shape
    pad = [(0, d % 2) for d in dims]
    w = np.pad(wall, pad)
    w = w.reshape(w.shape[0] // 2, 2, w.shape[1] // 2, 2, w.shape[2] // 2, 2).any(axis=(1, 3, 5))
    L = max(w.shape)
    enclosed = np.ones(w.shape, bool)
    for dx in (-1, 0, 1):
        for dy in (0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                hit, step = w.copy(), 1
                while step < L:                        # hit[v] = any(w[v + k*d] for k >= 0)
                    src, dst = [], []
                    for o in (dx * step, dy * step, dz * step):
                        src.append(slice(o, None) if o >= 0 else slice(None, o))
                        dst.append(slice(None, -o if o > 0 else None) if o >= 0 else slice(-o, None))
                    hit[tuple(dst)] |= hit[tuple(src)]
                    step *= 2
                enclosed &= hit
    up = enclosed.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    return up[:dims[0], :dims[1], :dims[2]]


def _diffuse_colours(faces: np.ndarray, col: np.ndarray, known: np.ndarray, max_iters: int = 2000) -> np.ndarray:
    """Fill unknown vertex colours with the mean of their neighbours, repeatedly (fronts move one ring per step)."""
    from scipy.sparse import coo_matrix
    nv = len(col)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    A = coo_matrix((np.ones(2 * len(e), np.float32), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                   shape=(nv, nv)).tocsr()
    col, known = col.copy(), known.copy()
    for _ in range(max_iters):
        if known.all():
            break
        w = A @ known.astype(np.float32)
        s = A @ (col * known[:, None])
        new = ~known & (w > 0)
        if not new.any():
            break
        col[new] = s[new] / w[new, None]
        known |= new
    return col
