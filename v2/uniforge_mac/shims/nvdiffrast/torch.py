"""Placeholder: nvdiffrast is only used for UV texture baking, which the Mac port replaces."""


def _na(*_a, **_k):
    raise NotImplementedError("nvdiffrast is not available on Mac; the port exports vertex colours instead")


RasterizeCudaContext = rasterize = interpolate = _na
