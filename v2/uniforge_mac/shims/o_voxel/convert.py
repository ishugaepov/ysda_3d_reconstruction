"""Flexible dual grid -> triangle mesh (pure torch).

Each sparse voxel carries a dual vertex (offset inside the voxel), 3 flags telling whether the
voxel's +x/+y/+z edges cross the surface, and a quad split weight. Every flagged edge is shared by
4 voxels; if all 4 exist they form a quad, split into two triangles along the diagonal with the
larger product of split weights (or the better-aligned normals if no weights).
"""
import torch

from flex_gemm.ops.spconv import lookup

# For an edge along axis a starting at voxel v, the 4 voxels sharing it (TRELLIS.2 order).
_EDGE_VOXELS = [
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],   # x edge
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],   # y edge
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],   # z edge
]
_SPLIT_A = [0, 1, 2, 0, 2, 3]
_SPLIT_B = [0, 1, 3, 3, 1, 2]


def flexible_dual_grid_to_mesh(coords, dual_vertices, intersected_flag, split_weight, aabb, voxel_size=None,
                               grid_size=None, train=False):
    dev = coords.device
    aabb = torch.as_tensor(aabb, dtype=torch.float32, device=dev)
    if voxel_size is not None:
        vs = torch.as_tensor([voxel_size] * 3 if isinstance(voxel_size, float) else voxel_size,
                             dtype=torch.float32, device=dev)
        gs = ((aabb[1] - aabb[0]) / vs).round().long()
    else:
        gs = torch.as_tensor([grid_size] * 3 if isinstance(grid_size, int) else grid_size, device=dev).long()
        vs = (aabb[1] - aabb[0]) / gs
    W, H, D = [int(x) for x in gs]
    c = coords.long()
    keys = (c[:, 0] * H + c[:, 1]) * D + c[:, 2]
    sorted_keys, order = torch.sort(keys)
    n_idx, axis = intersected_flag.bool().nonzero(as_tuple=True)
    offs = torch.tensor(_EDGE_VOXELS, device=dev, dtype=torch.long)          # (3,4,3)
    quad_vox = c[n_idx][:, None, :] + offs[axis]                             # (M,4,3)
    qk = (quad_vox[..., 0] * H + quad_vox[..., 1]) * D + quad_vox[..., 2]
    inb = ((quad_vox >= 0) & (quad_vox < torch.tensor([W, H, D], device=dev))).all(-1)
    idx = lookup(sorted_keys, order, qk.reshape(-1)).reshape(-1, 4)
    quads = idx[(inb & (idx >= 0)).all(1)]
    verts = (c.float() + dual_vertices.float()) * vs + aabb[0]
    sa, sb = torch.tensor(_SPLIT_A, device=dev), torch.tensor(_SPLIT_B, device=dev)
    ta, tb = quads[:, sa].reshape(-1, 2, 3), quads[:, sb].reshape(-1, 2, 3)
    if split_weight is None:
        def align(t):
            p = verts[t]
            n0 = torch.cross(p[:, 0, 1] - p[:, 0, 0], p[:, 0, 2] - p[:, 0, 0], dim=-1)
            n1 = torch.cross(p[:, 1, 1] - p[:, 1, 0], p[:, 1, 2] - p[:, 1, 0], dim=-1)
            return (n0 * n1).sum(-1).abs()
        use_a = align(ta) > align(tb)
    else:
        sw = split_weight.reshape(-1)[quads]
        use_a = sw[:, 0] * sw[:, 2] > sw[:, 1] * sw[:, 3]
    tris = torch.where(use_a[:, None, None], ta, tb).reshape(-1, 3)
    return verts, tris.int()
