"""Workarounds for ComfyUI behaviour on Apple Silicon. Only imported inside a ComfyUI process."""
from contextlib import contextmanager


@contextmanager
def memory_efficient_da3_attention():
    """Run DA3 with ComfyUI's sub-quadratic attention instead of attention_basic.

    On MPS, ComfyUI hands DINOv2 (DA3's backbone) and DA3's camera head `attention_basic` for
    "small inputs", which materialises the full (tokens x tokens) attention matrix. Multi-view DA3
    attends across all views jointly, so 21 views at 504 px (~27k tokens) needs a ~33 GB matrix
    and the process dies. attention_sub_quad computes the same result in chunks.
    """
    import comfy.image_encoders.dino2 as dino2
    import comfy.ldm.depth_anything_3.camera as camera
    from comfy.ldm.modules import attention

    def efficient(device, mask=False, small_input=False):
        return attention.attention_sub_quad

    saved = (dino2.optimized_attention_for_device, camera.optimized_attention_for_device)
    dino2.optimized_attention_for_device = efficient
    camera.optimized_attention_for_device = efficient
    try:
        yield
    finally:
        dino2.optimized_attention_for_device, camera.optimized_attention_for_device = saved
