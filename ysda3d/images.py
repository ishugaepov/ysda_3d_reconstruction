"""Loading arbitrary photo folders into one fixed-size batch."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def list_images(folder: str) -> list[str]:
    names = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(IMAGE_EXTS) and not f.startswith("."))
    if not names:
        raise ValueError(f"no images ({', '.join(IMAGE_EXTS)}) in {folder}")
    return [os.path.join(folder, f) for f in names]


def load_resized(path: str, max_side: int) -> np.ndarray:
    """RGB float32 in [0,1], EXIF orientation applied, longest side == max_side (never upscaled)."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = min(1.0, max_side / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def pad_to_batch(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Centre each image on a common canvas (max H x max W).

    Returns (B,H,W,3) images and a (B,H,W) mask that is 1 on real pixels and 0 on padding.
    Padding is needed because photos of one capture can be portrait and landscape.
    """
    H = max(im.shape[0] for im in images)
    W = max(im.shape[1] for im in images)
    batch = np.zeros((len(images), H, W, 3), np.float32)
    valid = np.zeros((len(images), H, W), np.float32)
    for i, im in enumerate(images):
        y0 = (H - im.shape[0]) // 2
        x0 = (W - im.shape[1]) // 2
        batch[i, y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        valid[i, y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = 1.0
    return batch, valid


def load_folder(folder: str, max_side: int = 1024) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = list_images(folder)
    batch, valid = pad_to_batch([load_resized(p, max_side) for p in paths])
    return batch, valid, [os.path.basename(p) for p in paths]


def rgba_views_from_masks(folder: str, names: list[str], masks, valid, out_dir: str, picks: list[int]) -> list[str]:
    """Write full-resolution RGBA PNGs (photo + mask alpha) for the picked views, numbered in order.

    masks/valid are the (B,H,W) arrays on the padded canvas of load_folder(); the valid region is
    mapped back to each photo's EXIF-rotated full resolution, so the object keeps all its pixels.
    """
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(out_dir, f))
    paths = []
    for k, i in enumerate(picks):
        photo = ImageOps.exif_transpose(Image.open(os.path.join(folder, names[i]))).convert("RGB")
        ys, xs = np.nonzero(np.asarray(valid[i]) > 0.5)
        m = np.asarray(masks[i])[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        alpha = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize(photo.size, Image.BILINEAR)
        photo.putalpha(alpha)
        p = os.path.join(out_dir, f"{k + 1:03d}_{os.path.splitext(names[i])[0]}.png")
        photo.save(p)
        paths.append(p)
    return paths
