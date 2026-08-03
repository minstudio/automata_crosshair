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

## Expected dataset structure

`paths.py` defines `DATASET_ROOT`, which defaults to `./dataset` next to this README. Either place your data there or override it with each script's `--dataset` argument (or edit `paths.py`).

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

Controls per image: left-drag to draw/redraw the ROI box, `Enter`/`R` to run BiRefNet on the current box, `S` to save the mask and move to the next image, `D` to skip without saving, `Q` to quit. Already-masked artifacts are skipped unless `--overwrite` is passed.

Single image mode:

```bash
python hsi/segment_roi_birefnet.py --viewfinder path/to/RGBVIEWFINDER_x.png --output hsi/output/roi/lot1n1_mask.png
```

Note: mask filenames must be `<artifact>_mask.png` for the later steps to find them.

### Step 2 - HSI clustering inside the ROI

Clusters the HSI cube pixels under each verified mask (HDBSCAN on PCA-reduced, SNV-normalised spectra by default) and writes `<artifact>_clusters.png` maps.

```bash
python hsi/run_roi_clustering.py
```

Useful options: `--artifacts lot1n1 lot2n15` to process specific artifacts, `--dataset <path>` to override the dataset root, `--algorithm {hdbscan,kmeans,gmm,agglomerative,optics}`, `--superpixel` for SLIC superpixel clustering, `--merge-small 0.05` to merge tiny clusters, `--spatial-weight` / `--pixel-weight` to mix spatial or visual features into the spectra. Run with `-h` for the full list.

### Step 3 - Crosshair generation on the 3D mesh

Registers each cluster map onto the artifact's mesh (silhouette matching over PCA-selected viewpoints), scores surface flatness, and picks the best measurement points per cluster. Processes every artifact that has both a cluster map and an OBJ.

```bash
python photogrammetry/analyze_hsi_clusters_remote.py --all
python photogrammetry/analyze_hsi_clusters_remote.py --artifact lot1n1
```

Useful options: `--xrf-only` to skip Raman targets, `--min-crosshairs N` (clusters too small for N placements are skipped), `--rim-dist` / `--rim-angle` to keep XRF points away from rims, `--raman-gap` to control Raman point spacing, `--grid-size` for registration resolution, `--output <dir>` to redirect results.

Output per artifact in `photogrammetry/output/hsi_clusters/`:

```
<artifact>_xrf.ply                    # flatness heatmap + ranked XRF target squares
<artifact>_raman.ply                  # Raman targets
<artifact>_hsi_clusters_regions.ply   # cluster regions painted on the mesh
```

Open the PLYs in MeshLab / CloudCompare to inspect the targets.

`photogrammetry/analyze_hsi_clusters.py` is the original local variant of the same script (kept for reference); `analyze_hsi_clusters_remote.py` is the tuned entry point to use.

## Module map

- `paths.py`: central `DATASET_ROOT` configuration.
- `hsi/segment_roi_birefnet.py`: manual ROI + BiRefNet viewfinder segmentation (step 1).
- `hsi/run_roi_clustering.py`: ROI-masked HSI clustering (step 2).
- `hsi/process_all_hdbscan_pca_single.py`: HSI cube reading + HDBSCAN/PCA clustering internals.
- `hsi/auto_segment_cluster_single.py`: clustering algorithms shared by step 2.
- `hsi/hsi_3d_registration.py`: silhouette-based HSI-to-3D registration + flatness scoring.
- `hsi/test_pca_viewpoints.py`: PCA-driven viewpoint selection used during registration.
- `photogrammetry/analyze_hsi_clusters_remote.py`: crosshair generation entry point (step 3).
