"""Make RGBA inputs for UniForge3D from real photos + our BiRefNet masks (from a DA3 .npz run).

The masks live on the padded max-side canvas of images.load_folder; they are mapped back to each
photo's full resolution (EXIF-rotated), so the object keeps its full pixel detail when UniForge3D
crops it. Output files are numbered in the given order, so the first view is the anchor.

  python scripts/make_rgba_views.py outputs/toy_da3.npz data/toy/images out_dir IMG_4505 IMG_4509 ...
"""
import os
import sys

import numpy as np
from PIL import Image, ImageOps

npz, folder, out_dir, *views = sys.argv[1:]
d = np.load(npz)
names = [str(n) for n in d["names"]]
masks, valid = d["masks"], d["valid"]
os.makedirs(out_dir, exist_ok=True)
for k, v in enumerate(views):
    i = next(j for j, n in enumerate(names) if n.startswith(v))
    photo = ImageOps.exif_transpose(Image.open(os.path.join(folder, names[i]))).convert("RGB")
    ys, xs = np.nonzero(valid[i] > 0.5)                      # where the resized photo sits on the canvas
    m = masks[i][ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alpha = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize(photo.size, Image.BILINEAR)
    rgba = photo.copy()
    rgba.putalpha(alpha)
    out = os.path.join(out_dir, f"{k + 1:03d}_{os.path.splitext(names[i])[0]}.png")
    rgba.save(out)
    print(out, photo.size, "alpha>0.5: %.1f%%" % (100 * (np.asarray(alpha) > 127).mean()))
