# 03 — Literature and materials

The reading list is organised by pipeline stage. Each section says **what math it gives you** and **where it plugs into our prototype**. ★ marks the most important items. Venues and years are from memory. Search the titles on arXiv or Google Scholar; most papers are freely available.

## 0. Math prerequisites (you likely know most of this; listed so nothing is implicit)

| Topic | Why it matters here | Material |
|---|---|---|
| Projective geometry, homogeneous coordinates, pinhole camera $x \sim K[R\mid t]X$ | Everything | ★ Hartley & Zisserman, ch. 2–6 |
| Rotations and rigid motions as Lie groups SO(3)/SE(3); tangent-space parameterisation | Optimising poses (bundle adjustment), averaging rotations | ★ Solà, Deray, Atchuthan, *A micro Lie theory for state estimation in robotics* (2018); Barfoot, *State Estimation for Robotics* (2nd ed., 2024) |
| Nonlinear least squares: Gauss–Newton, Levenberg–Marquardt, sparsity, the Schur complement, robust losses (Huber, Cauchy) | Bundle adjustment, pose refinement | ★ Triggs et al., *Bundle Adjustment — A Modern Synthesis* (1999); Nocedal & Wright, *Numerical Optimization* |
| Robust estimation: RANSAC and its variants | Two-view geometry, outlier rejection | Fischler & Bolles (1981); Raguram et al., USAC (2013); Barath & Matas, MAGSAC (2019) |
| Signed distance functions, level sets, the Poisson equation | Surface reconstruction | Osher & Fedkiw, *Level Set Methods*; Kazhdan papers (§4) |
| Volume rendering integral, alpha compositing | NeRF / Gaussian splatting | Max, *Optical Models for Direct Volume Rendering* (1995) |
| Discrete optimisation: MRFs, graph cuts, α-expansion | Texture view selection, MVS surface extraction | Boykov, Veksler, Zabih (2001); Kolmogorov & Zabih (2004) |
| Submodular maximisation | Next-best-view selection | ★ Krause & Golovin, *Submodular Function Maximization* (survey, 2014); Nemhauser, Wolsey, Fisher (1978) |

## 1. Textbooks and courses (backbone)

- ★ R. Hartley, A. Zisserman — *Multiple View Geometry in Computer Vision*, 2nd ed. (2004). The reference for epipolar geometry, triangulation, self-calibration and bundle adjustment.
- ★ R. Szeliski — *Computer Vision: Algorithms and Applications*, 2nd ed. (2022). Free PDF from the author. Chapters on SfM, dense reconstruction, 3D modelling and image-based rendering.
- Y. Ma, S. Soatto, J. Košecká, S. Sastry — *An Invitation to 3-D Vision* (2004). Rigorous, Lie-group flavoured geometric vision.
- W. Förstner, B. Wrobel — *Photogrammetric Computer Vision* (2016). Statistical and uncertainty view of photogrammetry, useful for NBV.
- Lecture series:
  - **Daniel Cremers, Multiple View Geometry** (TUM, on YouTube);
  - **Cyrill Stachniss, Photogrammetry I/II** (Bonn, on YouTube);
  - **Andreas Geiger, Computer Vision** (Tübingen; covers SfM, stereo, neural fields);
  - CMU 16-822 *Geometry-based Methods in Vision*.

## 2. Features, matching, structure from motion (stage B, classical)

- D. Lowe — *Distinctive Image Features from Scale-Invariant Keypoints* (IJCV 2004). SIFT.
- D. Nistér — *An Efficient Solution to the Five-Point Relative Pose Problem* (TPAMI 2004).
- N. Snavely, S. Seitz, R. Szeliski — *Photo Tourism* (SIGGRAPH 2006). Incremental SfM from unordered photos.
- ★ J. L. Schönberger, J.-M. Frahm — *Structure-from-Motion Revisited* (CVPR 2016). This is **COLMAP**, the de-facto standard; read it alongside the COLMAP docs.
- L. Pan et al. — *Global Structure-from-Motion Revisited* (ECCV 2024). GLOMAP: global rotation/translation averaging instead of incremental SfM.
- S. Agarwal et al. — *Bundle Adjustment in the Large* (ECCV 2010).
- Learned matching:
  - DeTone et al., *SuperPoint* (2018);
  - Sarlin et al., *SuperGlue* (CVPR 2020);
  - Sun et al., *LoFTR* (CVPR 2021);
  - Lindenberger et al., *LightGlue* (ICCV 2023).
- Software: COLMAP / pycolmap docs; *hloc* (hierarchical localization toolbox).

**Plugs into:** v2, refining DA3 poses with bundle adjustment.

## 3. Learned feed-forward 3D geometry (stage B in our prototype)

- ★ S. Wang et al. — *DUSt3R: Geometric 3D Vision Made Easy* (CVPR 2024). Predicts point maps from image pairs; the start of this line of work.
- V. Leroy et al. — *Grounding Image Matching in 3D with MASt3R* (ECCV 2024); also *MASt3R-SfM* (2024).
- ★ J. Wang et al. — *VGGT: Visual Geometry Grounded Transformer* (CVPR 2025, best paper). One transformer that returns cameras, depth, point maps and tracks for N views.
- **Depth Anything 3** (ByteDance Seed, 2025), *"Recovering the Visual Space from Any Views"*. This is the model we run; it predicts depth plus ray maps from any number of views. Read the paper for its depth-ray representation and how the camera head works.
- *MapAnything* (Meta, 2025). Universal metric reconstruction that can take optional intrinsics or poses.
- *π³: Permutation-Equivariant Visual Geometry Learning* (2025). Removes the need for a reference view.

**Plugs into:** Steps 2–3. These papers explain what DA3's confidence means, and why the poses are only approximate.

## 4. Dense reconstruction and surfaces (stage C)

- ★ Y. Furukawa, C. Hernández — *Multi-View Stereo: A Tutorial* (Foundations and Trends, 2015). The best single introduction to MVS.
- S. Seitz et al. — *A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms* (CVPR 2006).
- Y. Furukawa, J. Ponce — *Accurate, Dense, and Robust Multiview Stereopsis* (PMVS, TPAMI 2010).
- ★ J. L. Schönberger et al. — *Pixelwise View Selection for Unstructured Multi-View Stereo* (ECCV 2016). PatchMatch MVS, as in COLMAP.
- Y. Yao et al. — *MVSNet* (ECCV 2018). Learned MVS with cost volumes.
- ★ B. Curless, M. Levoy — *A Volumetric Method for Building Complex Models from Range Images* (SIGGRAPH 1996). **TSDF fusion**, which is our Step 4.
- R. Newcombe et al. — *KinectFusion* (ISMAR 2011). Real-time TSDF fusion.
- W. Lorensen, H. Cline — *Marching Cubes* (SIGGRAPH 1987).
- ★ M. Kazhdan, M. Bolitho, H. Hoppe — *Poisson Surface Reconstruction* (SGP 2006); M. Kazhdan, H. Hoppe — *Screened Poisson Surface Reconstruction* (TOG 2013).
- M. Jancosek, T. Pajdla — *Multi-View Reconstruction Preserving Weakly-Supported Surfaces* (CVPR 2011). Directly relevant to the vanishing laptop screen; the method behind OpenMVS meshing.
- Software: Open3D tutorials (TSDF, Poisson); OpenMVS wiki.

## 5. Texturing (stage D)

- ★ M. Waechter, N. Moehrle, M. Goesele — *Let There Be Color! Large-Scale Texturing of 3D Reconstructions* (ECCV 2014). MRF view selection plus global and local seam leveling; implemented in *mvs-texturing*, and the basis of OpenMVS's texturing.
- V. Lempitsky, D. Ivanov — *Seamless Mosaicing of Image-Based Texture Maps* (CVPR 2007).
- Q.-Y. Zhou, V. Koltun — *Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras* (SIGGRAPH 2014). Jointly optimises camera poses and colours; available in Open3D.
- B. Lévy et al. — *Least Squares Conformal Maps* (SIGGRAPH 2002). UV parameterisation theory; *xatlas* is the practical tool.

## 6. Neural / differentiable scene representations (later iterations, v7)

- ★ B. Mildenhall et al. — *NeRF* (ECCV 2020); J. Barron et al. — *Mip-NeRF 360* (CVPR 2022).
- P. Wang et al. — *NeuS* (NeurIPS 2021); Z. Li et al. — *Neuralangelo* (CVPR 2023). Signed-distance neural surfaces, giving better meshes than NeRF.
- ★ B. Kerbl et al. — *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023).
- B. Huang et al. — *2D Gaussian Splatting for Geometrically Accurate Radiance Fields* (SIGGRAPH 2024).
- A. Guédon, V. Lepetit — *SuGaR* (CVPR 2024); Z. Yu et al. — *Gaussian Opacity Fields* (2024); *PGSR* (2024). Ways to turn splats into meshes.
- gsplat (nerfstudio): its documentation includes a clean mathematical write-up of the splatting forward and backward passes.

## 7. Segmentation (stage A)

- P. Zheng et al. — *BiRefNet: Bilateral Reference for High-Resolution Dichotomous Image Segmentation* (2024).
- A. Kirillov et al. — *Segment Anything* (ICCV 2023); N. Ravi et al. — *SAM 2* (2024); *SAM 3* (Meta, 2025), with text-prompted concept segmentation.

## 8. Generative 3D (optional stage F)

- R. Liu et al. — *Zero-1-to-3* (ICCV 2023). Novel-view diffusion.
- J. Xiang et al. — *Structured 3D Latents for Scalable and Versatile 3D Generation* (TRELLIS, 2024), and its successor TRELLIS.2 (2025).
- Tencent — *Hunyuan3D 2.0* technical report (2025).

**Use with care:** generated geometry has to be registered to the reconstruction, and labelled as generated.

## 9. Next-best-view / active reconstruction (extension, stage E)

- C. Connolly — *The Determination of Next Best Views* (ICRA 1985).
- ★ W. Scott, G. Roth, J.-F. Rivest — *View Planning for Automated Three-Dimensional Object Reconstruction and Inspection* (ACM Computing Surveys, 2003).
- S. Isler et al. — *An Information Gain Formulation for Active Volumetric 3D Reconstruction* (ICRA 2016).
- ★ A. Krause, D. Golovin — *Submodular Function Maximization* (2014), for greedy guarantees.
- X. Pan et al. — *ActiveNeRF* (ECCV 2022); ★ W. Jiang, B. Lei, K. Daniilidis — *FisherRF: Active View Selection and Uncertainty Quantification for Radiance Fields using Fisher Information* (ECCV 2024).
- A product reference worth studying: Apple Object Capture's and RealityScan's in-app coverage guides.

## 10. Datasets for evaluation

- **Google Scanned Objects** (Downs et al., 2022). Real scanned household objects with meshes; good for synthetic renders with ground truth.
- **DTU MVS** (Aanæs et al., 2016) and **Tanks and Temples** (Knapitsch et al., 2017). Standard MVS benchmarks.
- **CO3D** (Reizenstein et al., 2021). Phone videos orbiting real objects, with poses and masks; close to our use case.
- **OmniObject3D** (Wu et al., 2023).

## Suggested reading order (about 3–4 weeks part-time)

1. Szeliski ch. on SfM + Hartley & Zisserman ch. 9–12 + Triggs BA → understand cameras and bundle adjustment.
2. Schönberger & Frahm (COLMAP) → DUSt3R → VGGT → Depth Anything 3 → see how learned models replace each classical block.
3. Furukawa–Hernández MVS tutorial → Curless & Levoy → Kazhdan (Poisson) → Jancosek & Pajdla → this is our surface stage.
4. Waechter et al. (texturing) → Zhou & Koltun → this is our texture stage.
5. Scott et al. survey → Krause & Golovin → FisherRF → the NBV extension.
6. 3DGS → 2DGS → SuGaR/GOF → for when we try a splat-based appearance model.
