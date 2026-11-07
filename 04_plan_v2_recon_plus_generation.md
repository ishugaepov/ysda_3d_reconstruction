# 04 — Plan v2: reconstruction + generation (UniForge3D direction)

*Written 2026-11-07 after reading UniForge3D (Chen et al., arXiv 2610.15357) and its code. Supersedes the
iteration backlog in 02_prototype_plan.md for the main track; v0/v1 stay as baselines.*

## 1. What UniForge3D does, and why it fits our goal

**Problem it solves:** sparse, unposed photos → a *complete* 3D asset that is *faithful* to what the photos show.
- Pure reconstruction (our v0/v1) is faithful but incomplete: holes where nothing was seen (the toy's underside), and degradation where views disagree.
- Pure generation (TRELLIS.2, Hunyuan3D) is complete but not faithful: it invents plausible details.

**Method (inference view):**
1. **Shared anchor-guided space.** The output frame is *upright* (canonical vertical axis) and its azimuth follows the **anchor view** (the first image). Reconstruction (VGGT point maps) and generation (TRELLIS.2 sparse structure) live in this same space, so no Sim(3) alignment between them is needed. As a bonus, the model comes out upright, which our v0/v1 can't do.
2. **Stage 1, joint model:**
   - a VGGT branch (LoRA) and the TRELLIS.2 sparse-structure DiT (LoRA) exchange features in both directions at 8 layer pairs;
   - reconstruction → generation: VGGT features become a residual added to DINOv3 tokens, used as cross-attention conditions;
   - generation → reconstruction: VGGT tokens cross-attend to DiT tokens, gated by the noise level;
   - outputs: per-view **point maps** and a complete **64³ sparse structure**.
3. **Stage 2, structure-guided reconstruction plus voxel-wise view selection:**
   - a second VGGT, conditioned on the generated structure, predicts point maps in the same space, which gives **pixel↔voxel correspondences**;
   - per voxel and view, a support score s_v(x) = size of the largest connected pixel component mapping into that voxel;
   - each voxel is routed to v* = argmax_v s_v(x); voxels with no support inherit from their 26-neighbourhood, else the anchor view;
   - the shape DiT (512 → 1024) alternates **global multi-view attention** with **selected-view attention**.
4. **Stage 3, texture:** the same routing, then a TRELLIS.2 texture DiT; baked into PBR UV textures.

**Results:**
- On GSO (4 views) it is state of the art: F-score 0.89, against 0.65–0.74 for Hunyuan3D-2mv, ReconViaGen, Mix3R and MV-SAM3D.
- Ablations: the joint space beats canonical and cross-space alternatives (IoU 0.58 vs 0.46); structure-conditioned VGGT gives better point maps; voxel routing gives better detail.

**Stated limitation:** trained on synthetic renders; *"severe occlusion, strong reflections, and extreme lighting remain challenging"*. Real photos are exactly where we come in.

**Why it fits our constraints:**
- No training needed: weights are released.
- It accepts an **arbitrary number of unposed views**; the code batches all images; training used 4.
- Its per-voxel support score s_v(x) is a ready-made **observation-quality signal for next-best-view**: voxels nobody supports are generated (hallucinated), not reconstructed.

## 2. Feasibility on our Mac (checked in the code, not yet run)

| Item | Finding | Plan |
|---|---|---|
| Weights | `xibushu/UniForge3D`, 27 GB, **CC BY-NC 4.0** (fine for study/research); loaded with `torch.load(weights_only=True)` (safe) | downloaded |
| TRELLIS.2 parts | `microsoft/TRELLIS.2-4B` (3 ckpts, pinned rev), `microsoft/TRELLIS-image-large` ss decoder, MIT | downloaded |
| DINOv3-L | `facebook/dinov3-vitl16-pretrain-lvd1689m` is **gated (manual approval by Meta)** | either the user requests access + HF token, or convert ComfyUI's ungated `dino_v3_vit_l.safetensors` |
| Background removal | RMBG-2.0 (gated, NC) is loaded eagerly, but **RGBA inputs with real alpha skip it** | feed our BiRefNet masks as RGBA; patch out the eager load |
| Device | hard-coded `torch.device("cuda")`, `torch.cuda.*`, `autocast("cuda")` | small patches: device = mps/cpu |
| Attention | flash-attn default, but `sdpa` / `naive` backends exist | `ATTN_BACKEND=sdpa` |
| Sparse conv | only `flex_gemm` (CUDA) | swap in ComfyUI's pure-torch `comfy/ldm/trellis2/flexgemm.py` (or a minimal torch implementation) |
| Mesh extraction | `o_voxel.convert.flexible_dual_grid_to_mesh` (CUDA ext) | ComfyUI's TRELLIS.2 VAE decode (`comfy/ldm/trellis2/vae.py`) |
| Post-process / bake | CuMesh (remesh, BVH), nvdiffrast | ComfyUI's mesh3d post-processing + `BakeTextureFromVoxel`, or our own projection baker |
| Memory | VGGT ×2 (~4–5 GB each), DiT 1.3B + three ~2.7B DiTs, DINOv3-L; `low_vram` loads stage by stage | fits in 64 GB unified memory, sequentially; speed unknown (MPS, no flash-attn) |

**Risk:** the port is real engineering (CUDA → torch shims). The fallback is a rented CUDA GPU just for this stage, but that breaks the "on the laptop" goal, so we'd ask first.

**Second route that needs no port** (inspired by ReconPlusGen and UMI3D, both training-free in spirit): use ComfyUI's **native** TRELLIS.2, and inject *our* reconstruction as its sparse structure (see Track B).

## 3. Learnings from v0–v1 that shape the plan

1. Masks (BiRefNet) are reliable on this kind of object, so keep them.
2. **COLMAP cameras are accurate** (0.7 px), and they are what make photo-faithful colour possible. DA3 poses are not good enough for texturing (journal 0004/0005).
3. DA3 depth is ~1.4% inconsistent across views, so don't let it decide fine geometry.
4. A strict visual hull with good cameras matches the masks (IoU 0.956) but can't do concavities, and colours patch when visibility relies on DA3 depth. **Visibility must come from the mesh itself** (z-buffer).
5. Real captures break assumptions: mixed lenses, EXIF rotation, exposure changes, wrong COLMAP registrations, ground-level views that don't register. Every stage needs a sanity check.
6. Measure, don't eyeball: silhouette IoU, cross-view consistency, render-vs-photo sheets, and side-by-side renders against KIRI/Polycam.

## 4. Architecture v2

```
photos (arbitrary N, any angles)
  └─ A. masks (BiRefNet) + EXIF/lens handling ............................. existing
  └─ B. COLMAP cameras (v1) on ALL photos .................................. existing
        → accurate K, R, t, k1 for every registered photo; "gravity" from the camera up-vectors/floor plane
  └─ G. generator, given K selected views (masked RGBA, object-centred crops):
        G1 = UniForge3D (ported)  |  G2 = native TRELLIS.2 + our structure (Track B)  |  G3 = Pixal3D (baseline)
        → complete mesh (+ generated texture) in the generator's frame
  └─ R. register generator frame → COLMAP frame
        UniForge3D: its per-view point maps give pixel↔3D correspondences for the input views →
        PnP / Umeyama against our cameras; refine by silhouette alignment over ALL photos
  └─ T. appearance: re-texture every surface point SEEN by the photos (best view, mesh z-buffer
        visibility, exposure compensation); keep the generated texture only where nothing was seen;
        mark which is which
  └─ E. evaluation on photos NOT fed to the generator (held-out): silhouette IoU, masked PSNR/LPIPS/ΔE
  └─ N. next-best-view: per-surface support (photo coverage + the generator's routing support) →
        greedy candidate views on the upright sphere → previews rendered from the COMPLETE generated mesh
```

The key idea: **generation supplies completeness and an upright frame; our calibrated photos supply faithfulness and the ground truth for evaluation**. The photos not fed to the generator become a free held-out test set.

## 5. Iterations (each one gets a journal entry)

### v2.0 — Feasibility on the Mac (gate for everything else)
1. Run ComfyUI's **native TRELLIS.2 single-image template** on one toy photo: does the sparse stack run on MPS, and how long and how much memory?
2. Run the **Pixal3D multi-view** template on 4 toy views (baseline G3; it expects turnaround-style views).
3. **Port UniForge3D inference** to MPS with the shims from §2, in `third_party_patches/`, without modifying weights. Validate on the authors' `assets/case01` (4 clean renders) for visual sanity, then on the toy.
4. Get DINOv3 weights (user access request, or conversion of the ComfyUI repack; check that features match).

**Exit criterion:** a UniForge3D GLB of the toy produced on the laptop, or a documented blocker plus a decision (Track B only, or a GPU rental).

### v2.1 — Real-photo input adapter
- RGBA from BiRefNet; per-view square crop (the code does this); EXIF rotation.
- **View selection for K inputs** from N photos (the model was trained with 4). Use COLMAP poses to pick an anchor (most frontal/informative: biggest mask, sharp, upright) plus K−1 views maximising angular spread; skip blurry or badly exposed ones.
- Ablate K = 4 / 6 / 8 / all, and the choice of anchor.

### v2.2 — Registration and held-out evaluation
- Sim(3) from generator space to the COLMAP frame, via point maps or silhouettes; verify with silhouette IoU on **held-out** photos.
- Metrics table: v1 vs UniForge3D vs Track B vs Pixal3D vs KIRI/Polycam, on held-out photos.

### v2.3 — Faithful appearance
- Photo re-texturing of observed regions (best view per texel; mesh z-buffer visibility; per-image gain).
- Generated texture only for unobserved regions; visualise the "observed vs generated" mask.

### v2.4 — Next-best-view (the extension)
- Support per surface element: (a) photo coverage (number of views, incidence angle, resolution, baseline), and (b) the generator's voxel routing support s_v(x).
- Candidate cameras on the upright view-sphere; greedy area-weighted gain (coverage is submodular).
- Output in ComfyUI: a heatmap mesh, top-k viewpoint frustums, and **preview renders of the complete generated model from the suggested viewpoints**. Generation makes previews possible even for regions never photographed.
- Validate by "simulated capture": hold out photos, check whether NBV would have asked for the views that most improve held-out metrics.

### Track B (in parallel, no port needed) — native TRELLIS.2 with our structure
- Voxelise our v1 shape (visual hull ∩ depth carve, COLMAP cameras) into TRELLIS.2's 64³ sparse structure in an upright, normalised frame.
- Skip TRELLIS.2's structure stage; run its shape and texture stages conditioned on the selected photos. Optionally, multi-image conditioning via per-voxel routing in the spirit of UMI3D (training-free).
- **Compare against UniForge3D**: if comparable, it's the simplest robust option on the Mac.

## 6. Datasets for v2
- `data/toy` (21 photos; ground-level shots and the underside are the hard parts).
- 2–3 new real captures, including one hard case (the laptop: thin, reflective, low texture).
- The authors' `assets/case01–18` (clean renders) for sanity-checking the port.
- GSO renders composited onto real backgrounds (with known geometry) for quantitative checks; plan item 7 of 02.

## 7. Reading list additions (see 03_literature.md)
UniForge3D; TRELLIS.2 (Native and Compact Structured Latents); VGGT; DINOv3; ReconViaGen; Mix3R; RecGen3D; ReconPlusGen; UMI3D; ROAR-3D; MV-SAM3D; ShapeR (casual captures); Pixal3D.
