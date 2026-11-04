# XRF / Raman Crosshair Generation

Pipeline that places XRF and Raman measurement targets (crosshairs) on 3D artifact meshes, driven by HSI spectral clustering. For each artifact it: segments the artifact in the HSI viewfinder, clusters the HSI cube into spectrally distinct regions, registers the 2D cluster map onto the photogrammetry mesh via silhouette matching, and selects the flattest non-overlapping measurement positions per cluster. Output is a set of PLY files with the crosshair positions.

## Installation

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Notes:
- `torch` / `torchvision` with CUDA is strongly recommended, since BiRefNet segmentation and mesh rendering are slow on CPU. Install per https://pytorch.org/get-started/locally/ for your CUDA version.
- `pyrender` needs OpenGL. On headless Linux set `PYOPENGL_PLATFORM=egl` (the scripts attempt this automatically).
- BiRefNet weights (`ZhengPeng7/BiRefNet`) are downloaded automatically from Hugging Face on first run.
- `gpu.py` picks the device (CUDA, ROCm or CPU) and runs BiRefNet in fp16 on GPU. Check the setup with `python gpu.py --birefnet`. Env vars: `AUTOMATA_DEVICE=cpu|cuda` to force a device, `AUTOMATA_FP16=0` for fp32 if masks look wrong, `AUTOMATA_NO_FALLBACK=1` to crash instead of falling back to CPU.

Windows shortcut: `run_crosshair_generation.bat` installs the mesh dependencies and runs step 3 for all artifacts.

## Expected dataset structure

`paths.py` defines `DATASET_ROOT`, which defaults to `./dataset` next to this README. Override it for all scripts with the `AUTOMATA_DATASET` environment variable, per script with `--dataset`, or edit `paths.py`.

One folder per artifact, named after the artifact (e.g. `lot1n1`). Only HSI and photogrammetry data are required:

```
dataset/
└── lot1n1/
    ├── HSI/
    │   └── raw_data/
    │       ├── capture/                     # raw acquisition
    │       │   ├── <name>.raw               # HSI cube (DARKREF/WHITEREF excluded automatically)
    │       │   └── <name>.hdr               # ENVI header
    │       ├── metadata/
    │       │   └── <name>.xml               # contains datacube_angle
    │       └── results/                     # camera-corrected exports (preferred when present)
    │           ├── REFLECTANCE_<name>.dat
    │           ├── REFLECTANCE_<name>.hdr
    │           ├── REFLECTANCE_<name>.png
    │           └── RGBVIEWFINDER_<name>.png # viewfinder image used for ROI selection
    └── photogrammetry/
        └── raw_data/
            └── <name>.obj                   # 3D mesh (+ .mtl / textures alongside)
```

The mesh is looked up in `photogrammetry/raw_data/` first, then `photogrammetry/processed_data/` (e.g. the pipeline's `textured_mesh.glb`). Formats: `.obj`, `.glb`/`.gltf`, `.ply`. Units (m, cm or mm) are detected from the mesh size; output points are in cm in the mesh's own frame.

`HSI/capture/` (without `raw_data`) is also accepted as a fallback.

Intermediate and final results live inside the repo:

```
hsi/output/roi/                        # step 1: ROI masks (<artifact>_mask.png + overlay)
hsi/output/roi_clustered/              # step 2: cluster maps (<artifact>_clusters.png, ...)
photogrammetry/output/hsi_clusters/    # step 3: crosshair PLYs
```

## Workflow

### Step 1 - Manual ROI selection + BiRefNet segmentation

Interactive: for each artifact you draw a bounding box around the object in the viewfinder image, BiRefNet segments it inside that box, and the mask is saved in full-image coordinates.

Always run batch mode with `--hsi_only`. Without it the script also requires legacy `pXRF/` and `Raman/` subfolders that this pipeline does not use:

```bash
python hsi/segment_roi_birefnet.py --batch --hsi_only --output_dir hsi/output/roi
```

Add `--artifacts lot1n1 lot2n15` to segment only specific artifacts.

Controls per image: left-drag to draw/redraw the ROI box, `Enter`/`R` to run BiRefNet on the current box, `S` to save the mask and move to the next image, `D` to skip without saving, `Q` to quit. Already-masked artifacts are skipped unless `--overwrite` is passed.

Single image mode:

```bash
python hsi/segment_roi_birefnet.py --viewfinder path/to/RGBVIEWFINDER_x.png --output hsi/output/roi/lot1n1_mask.png
```

Note: mask filenames must be `<artifact>_mask.png` for the later steps to find them.

### Step 2 - HSI clustering inside the ROI

Clusters the HSI cube pixels under each verified mask and writes `<artifact>_clusters.png` maps.

Default `--algorithm auto` (number of clusters found automatically):

1. Features per pixel: SNV spectral shape reduced to 5 PCA components, plus log brightness at half weight (`--intensity-weight`, default 0.5).
2. Pixels near region boundaries (mixed spectra) are excluded from the fit: only the half with the lowest spectral gradient is used (`--pure-quantile`, default 50).
3. HDBSCAN on those pixels decides the clusters; every other pixel is assigned by 15-nearest-neighbour vote in feature space.
4. By default the cube is the calibrated reflectance (`--source reflectance`, `--source raw` for the raw capture) with 3x3 spatial smoothing (`--smooth`).

```bash
python hsi/run_roi_clustering.py
```

Useful options: `--artifacts lot1n1 lot2n15` to process specific artifacts, `--dataset <path>` to override the dataset root, `--algorithm {auto,hdbscan,kmeans,gmm,agglomerative,optics,...}` (the non-auto ones use the previous pipeline: raw cube, no smoothing unless `--smooth` is given), `--superpixel` for SLIC superpixel clustering, `--merge-small 0.05` to merge tiny clusters, `--spatial-weight` / `--pixel-weight` to mix spatial or visual features into the spectra. Run with `-h` for the full list.

Also: `--source reflectance` clusters the camera-calibrated `REFLECTANCE_*.dat` instead of the raw capture, `--smooth 3` applies a 3x3 spatial mean before clustering. Clusters may span several separate regions (repeated decoration); only specks under 1% of the ROI are dropped and reassigned to the nearest cluster.

### Step 3 - Crosshair generation on the 3D mesh

Registers each cluster map onto the artifact's mesh (silhouette matching over PCA-selected viewpoints), scores surface flatness, and picks the best measurement points per cluster. XRF uses BFS flat-region search, restricted to the surface visible from the HSI viewpoint, with a soft rim penalty; Raman reuses XRF points that fall in a cluster and fills the rest at the flattest blob pixel away from rims and other clusters. `--all` processes every artifact that has both a cluster map and an OBJ.

```bash
python photogrammetry/analyze_hsi_clusters.py --all
python photogrammetry/analyze_hsi_clusters.py --artifact lot1n1
```

Useful options: `--xrf-only` to skip Raman targets, `--min-crosshairs N` (clusters with fewer viable targets are skipped), `--rim-dist` / `--rim-angle` to keep points away from rims (`--no-rim` to disable), `--cluster-gap-px` to keep Raman points away from other clusters, `--raman-core` (0.5) to keep Raman points in the interior of each cluster blob, `--xrf-share-cm` / `--no-xrf-share` for reusing XRF points as Raman targets, `--raman-centroid` for blob-centroid Raman placement, `--grid-size` for registration resolution, `--output <dir>` to redirect results.

Output per artifact in `photogrammetry/output/hsi_clusters/`:

```
<artifact>_xrf.ply                    # mesh texture (or flatness heatmap, --xrf-base flatness) + ranked XRF target squares
<artifact>_raman.ply                  # Raman targets
<artifact>_hsi_clusters_regions.ply   # cluster regions painted on the mesh
```

Both JSON reports give each target as `point` (cm, in the source mesh's frame), `normal` (unit outward surface normal in the same frame, area-weighted over faces within 0.35 cm, oriented toward the HSI viewpoint) and `vertex_idx` (index of the mesh vertex at the target).

Open the PLYs in MeshLab / CloudCompare to inspect the targets.

`photogrammetry/analyze_hsi_clusters_remote.py` is the previous remote version, kept for reference only.

## Module map

- `paths.py`: central `DATASET_ROOT` configuration.
- `hsi/segment_roi_birefnet.py`: manual ROI + BiRefNet viewfinder segmentation (step 1).
- `hsi/run_roi_clustering.py`: ROI-masked HSI clustering (step 2).
- `hsi/process_all_hdbscan_pca_single.py`: HSI cube reading + HDBSCAN/PCA clustering internals.
- `hsi/auto_segment_cluster_single.py`: clustering algorithms shared by step 2.
- `hsi/hsi_3d_registration.py`: silhouette-based HSI-to-3D registration + flatness scoring.
- `hsi/test_pca_viewpoints.py`: PCA-driven viewpoint selection used during registration.
- `photogrammetry/analyze_hsi_clusters.py`: crosshair generation entry point (step 3).
- `photogrammetry/analyze_hsi_clusters_remote.py`: previous remote version (legacy).
- `gpu.py`: device selection and BiRefNet loading/inference with CPU fallback.
