"""CPU stand-in for the CuMesh CUDA library: the subset UniForge3D uses outside texture baking.

Decimation = quadric edge collapse (fast-simplification). Cleaning ops are light trimesh versions;
hole filling and narrow-band remeshing are skipped (the Mac export path doesn't need them).
"""
import numpy as np
import torch
import trimesh


class CuMesh:
    def init(self, vertices, faces):
        self._dev = vertices.device if isinstance(vertices, torch.Tensor) else "cpu"
        self.v = np.asarray(vertices.detach().float().cpu() if isinstance(vertices, torch.Tensor) else vertices, np.float64)
        self.f = np.asarray(faces.detach().cpu() if isinstance(faces, torch.Tensor) else faces, np.int64)
        self.n = None

    def read(self):
        return (torch.from_numpy(self.v.astype(np.float32)).to(self._dev),
                torch.from_numpy(self.f.astype(np.int32)).to(self._dev))

    def simplify(self, target, verbose=False, options=None):
        if len(self.f) <= target:
            return
        import fast_simplification
        self.v, self.f = fast_simplification.simplify(self.v.astype(np.float32), self.f.astype(np.int32),
                                                      target_reduction=1.0 - target / len(self.f))
        self.v, self.f = self.v.astype(np.float64), self.f.astype(np.int64)

    def _tm(self):
        return trimesh.Trimesh(self.v, self.f, process=False)

    def _set(self, m):
        self.v, self.f = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)

    def remove_faces(self, mask):
        keep = ~np.asarray(mask.cpu() if isinstance(mask, torch.Tensor) else mask, bool)
        m = self._tm(); m.update_faces(keep); m.remove_unreferenced_vertices(); self._set(m)

    def remove_duplicate_faces(self):
        m = self._tm(); m.update_faces(m.unique_faces()); self._set(m)

    def remove_small_connected_components(self, min_area_frac=1e-5):
        m = self._tm()
        parts = m.split(only_watertight=False)
        if len(parts) > 1:
            total = m.area
            keep = [p for p in parts if p.area >= min_area_frac * total]
            self._set(trimesh.util.concatenate(keep))

    def fill_holes(self, max_hole_perimeter=None):
        pass  # skipped on Mac (journal 0009)

    def repair_non_manifold_edges(self):
        pass

    def unify_face_orientations(self):
        m = self._tm(); trimesh.repair.fix_winding(m); self._set(m)

    def compute_vertex_normals(self):
        self.n = np.asarray(self._tm().vertex_normals, np.float32)

    def read_vertex_normals(self):
        if self.n is None:
            self.compute_vertex_normals()
        return torch.from_numpy(self.n).to(self._dev)


class cuBVH:  # noqa: N801
    def __init__(self, *_a, **_k):
        raise NotImplementedError("cuBVH is only used by remeshing/baking, which the Mac port replaces")


class remeshing:  # noqa: N801
    @staticmethod
    def remesh_narrow_band_dc(*_a, **_k):
        raise NotImplementedError("narrow-band remeshing is replaced on Mac")
