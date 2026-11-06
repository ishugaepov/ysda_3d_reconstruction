"""COLMAP SfM worker. Runs in its own process because pycolmap and torch bundle conflicting OpenMP
runtimes on macOS (importing both into one process aborts, including inside the ComfyUI server).

Must NOT import torch. Usage:
  python -m ysda3d.sfm_worker <image_root> <work_dir> <out.npz> [camera_model]
Writes, for the largest reconstruction: per image name, K (3x3), camera params, world→camera 3x4,
and its observations (pixel xy + 3D point xyz) for points that were triangulated.
"""
import json
import os
import shutil
import sys

import numpy as np


def main(img_root, work_dir, out_path, camera_model="SIMPLE_RADIAL"):
    import pycolmap
    db = os.path.join(work_dir, "database.db")
    sparse = os.path.join(work_dir, "sparse")
    if os.path.exists(db):
        os.remove(db)
    shutil.rmtree(sparse, ignore_errors=True)
    os.makedirs(sparse)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    pycolmap.extract_features(db, img_root, camera_mode=pycolmap.CameraMode.PER_FOLDER, reader_options=reader)
    pycolmap.match_exhaustive(db)
    maps = pycolmap.incremental_mapping(db, img_root, sparse)
    if not maps:
        json.dump({"error": "no reconstruction"}, open(out_path + ".json", "w"))
        sys.exit(2)
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    names, Ks, Ms, obs_xy, obs_xyz, obs_img, cams = [], [], [], [], [], [], {}
    for k, im in enumerate(sorted(rec.images.values(), key=lambda x: x.name)):
        cam = rec.cameras[im.camera_id]
        pose = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
        names.append(im.name)
        Ks.append(np.array(cam.calibration_matrix()))
        Ms.append(np.array(pose.matrix()))
        cams[im.name] = {"camera_id": int(im.camera_id), "model": str(cam.model), "params": [float(x) for x in cam.params]}
        for p in im.points2D:
            if p.has_point3D():
                obs_xy.append(p.xy); obs_xyz.append(rec.points3D[p.point3D_id].xyz); obs_img.append(k)
    reproj = float(rec.compute_mean_reprojection_error()) if hasattr(rec, "compute_mean_reprojection_error") else None
    np.savez(out_path, names=np.array(names), K=np.array(Ks), M=np.array(Ms),
             obs_xy=np.array(obs_xy), obs_xyz=np.array(obs_xyz), obs_img=np.array(obs_img))
    json.dump({"reconstructions": {str(k): r.num_reg_images() for k, r in maps.items()},
               "points3D": len(rec.points3D), "mean_reprojection_error_px": reproj, "cameras": cams},
              open(out_path + ".json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
