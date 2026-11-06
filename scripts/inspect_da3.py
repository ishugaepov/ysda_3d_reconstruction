"""Sanity checks on a saved DA3 run (.npz from run_da3_offline.py): masks, cameras, intrinsics."""
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from ysda3d.recon import da3_processing_size  # noqa: E402

d = np.load(sys.argv[1])
out_prefix = sys.argv[2]
names, imgs, masks = d["names"], d["images"], d["masks"]
K, E = d["intrinsics"], d["extrinsics"]
S, H, W = masks.shape
ph, pw = da3_processing_size(H, W, int(d["da3_res"]))
print(f"{S} views, image {H}x{W}, DA3 processing {ph}x{pw}")

# Masked contact sheet: object in colour, background darkened.
cols, tw = 7, 220
sheet = Image.new("RGB", (cols * tw, ((S + cols - 1) // cols) * tw), "white")
for i in range(S):
    m = masks[i][..., None]
    vis = (imgs[i] * (0.25 + 0.75 * m) * 255).astype(np.uint8)
    im = Image.fromarray(vis).resize((tw, tw))
    sheet.paste(im, ((i % cols) * tw, (i // cols) * tw))
    ImageDraw.Draw(sheet).text(((i % cols) * tw + 4, (i // cols) * tw + 4), str(names[i])[4:8], fill="yellow")
sheet.save(out_prefix + "_masks.jpg", quality=85)

# Cameras: centres, distance to object centre, focal in 35mm-equivalent terms.
R, t = E[:, :3, :3], E[:, :3, 3]
C = -np.einsum("sji,sj->si", R, t)
fwd = R[:, 2, :]  # camera z axis in world
centre = C.mean(0)
print("name      mask%   f_x(proc px)  f35eq   dist   cam-height(-y)")
for i in range(S):
    f35 = K[i, 0, 0] / pw * 36.0 * (max(H, W) / W)  # 36 mm across the long side
    print(f"{str(names[i])[4:8]}  {100*masks[i].mean():6.1f}  {K[i,0,0]:10.1f}  {f35:6.1f}  "
          f"{np.linalg.norm(C[i]-centre):5.2f}  {-C[i,1]:+.2f}")

# Top-down plot of camera positions + viewing directions (x-z plane).
img = Image.new("RGB", (600, 600), "white")
dr = ImageDraw.Draw(img)
span = np.abs(C[:, [0, 2]] - centre[[0, 2]]).max() * 1.2
def to_px(p):
    return (300 + (p[0] - centre[0]) / span * 280, 300 + (p[2] - centre[2]) / span * 280)
for i in range(S):
    a = to_px(C[i]); b = to_px(C[i] + fwd[i] * span * 0.15)
    dr.line([a, b], fill="red", width=2); dr.ellipse([a[0]-4, a[1]-4, a[0]+4, a[1]+4], fill="blue")
    dr.text((a[0]+5, a[1]-5), str(names[i])[4:8], fill="black")
img.save(out_prefix + "_cameras_topdown.png")
print("saved", out_prefix + "_masks.jpg", out_prefix + "_cameras_topdown.png")
