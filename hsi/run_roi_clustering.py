"""
Run HDBSCAN clustering on artifacts with verified ROI masks.

Uses the clustering algorithm from hsi/auto_segment_cluster_single.py
with the client-provided masks from hsi/output/roi/ instead of auto-segmented ones.

Usage:
    python hsi/run_roi_clustering.py
    python hsi/run_roi_clustering.py --dataset <path> --roi <path> --output <path>
    python hsi/run_roi_clustering.py --artifacts lot1n1 lot2n15
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import DATASET_ROOT
from hsi.process_all_hdbscan_pca_single import read_hsi_cube
from hsi.hsi_3d_registration import parse_datacube_angle
from hsi.auto_segment_cluster_single import (
    run_hdbscan_clustering, run_clustering, run_superpixel_clustering,
    find_capture_path, find_hsi_files,
)


def get_datacube_angle(artifact_folder):
    """Return datacube_angle from the HSI metadata XML, or 0 if not found."""
    import glob
    pattern = os.path.join(artifact_folder, "HSI", "raw_data", "metadata", "*.xml")
    xmls = glob.glob(pattern)
    return parse_datacube_angle(xmls[0] if xmls else None)


def rotate_mask_to_cube(mask_u8, angle):
    """Rotate mask from viewfinder space into cube space (mirrors test_2d_to_3d logic)."""
    if angle == 0:
        return mask_u8
    k = ((-angle) // 90) % 4
    return np.rot90(mask_u8, k=k).copy()


def find_reflectance_png(artifact_folder):
    """Return path to REFLECTANCE_*.png in HSI/raw_data/results/, or None."""
    results_dir = os.path.join(artifact_folder, "HSI", "raw_data", "results")
    if not os.path.isdir(results_dir):
        return None
    for f in os.listdir(results_dir):
        if f.startswith("REFLECTANCE_") and f.endswith(".png"):
            return os.path.join(results_dir, f)
    return None


def _mask_bbox(mask_u8, pad=20):
    ys, xs = np.where(mask_u8 > 127)
    if len(ys) == 0:
        return None
    h, w = mask_u8.shape
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + pad),
        min(h, int(ys.max()) + pad),
    )


def _draw_mask_boundary(img_array, mask_2d, color=(0, 255, 0)):
    from scipy import ndimage
    boundary = mask_2d & ~ndimage.binary_erosion(mask_2d, iterations=1)
    img_array[boundary] = color


def save_cropped_with_outline(reflectance_path, mask_u8, output_path, pad=20):
    """Crop REFLECTANCE_*.png to mask bounding box with green mask boundary."""
    rgb = Image.open(reflectance_path).convert("RGB")
    mh, mw = mask_u8.shape
    if rgb.size != (mw, mh):
        rgb = rgb.resize((mw, mh), Image.LANCZOS)

    bbox = _mask_bbox(mask_u8, pad)
    if bbox is None:
        rgb.save(output_path)
        return
    x0, y0, x1, y1 = bbox

    arr = np.array(rgb.crop((x0, y0, x1, y1)))
    mask_crop = mask_u8[y0:y1, x0:x1] > 127
    _draw_mask_boundary(arr, mask_crop)
    Image.fromarray(arr).save(output_path)


def save_cluster_overlay_crop(reflectance_path, hdbscan_png_path, mask_u8,
                               output_path, alpha=0.55, pad=20, cube_angle=0):
    """
    Blend cluster colours (from hdbscan_pca.png) onto the REFLECTANCE image,
    draw the mask boundary, and crop to the mask bounding box.
    hdbscan_pca.png is in cube space; cube_angle rotates it back to viewfinder space.
    """
    rgb = np.array(Image.open(reflectance_path).convert("RGB"))
    mh, mw = mask_u8.shape
    if rgb.shape[:2] != (mh, mw):
        rgb = np.array(Image.fromarray(rgb).resize((mw, mh), Image.LANCZOS))

    clusters = np.array(Image.open(hdbscan_png_path).convert("RGB"))
    # rotate clusters from cube space back to viewfinder/REFLECTANCE-PNG space
    if cube_angle != 0:
        k_inv = (4 - int(((-cube_angle) // 90) % 4)) % 4
        clusters = np.rot90(clusters, k=k_inv).copy()

    # Blend only where there is a cluster colour (non-black pixel)
    colored = clusters.sum(axis=2) > 0
    out = rgb.copy()
    out[colored] = (
        rgb[colored].astype(np.float32) * (1 - alpha)
        + clusters[colored].astype(np.float32) * alpha
    ).astype(np.uint8)

    _draw_mask_boundary(out, mask_u8 > 127)

    bbox = _mask_bbox(mask_u8, pad)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        out = out[y0:y1, x0:x1]

    Image.fromarray(out).save(output_path)

DEFAULT_ROI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "roi")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "roi_clustered")


def main():
    parser = argparse.ArgumentParser(description="Cluster HSI artifacts using verified ROI masks")
    parser.add_argument("--dataset", default=str(DATASET_ROOT), help="Dataset root directory")
    parser.add_argument("--roi", default=DEFAULT_ROI_DIR, help="Directory containing verified *_mask.png files")
    parser.add_argument("--output", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--artifacts", nargs="*", help="Specific artifact names to process (default: all with ROI mask)")
    parser.add_argument("--min-cluster-size", type=int, default=None, help="HDBSCAN/OPTICS min_cluster_size (default: auto)")
    parser.add_argument("--min-samples", type=int, default=None, help="HDBSCAN/OPTICS min_samples (default: auto)")
    parser.add_argument("--epsilon", type=float, default=0.3, help="HDBSCAN cluster_selection_epsilon (default: 0.3)")
    parser.add_argument("--method", default="eom", choices=["eom", "leaf"], help="HDBSCAN cluster_selection_method (default: eom)")
    parser.add_argument("--algorithm", default="hdbscan",
                        choices=["hdbscan", "kmeans", "gmm", "agglomerative", "optics",
                                 "meanshift", "bgmm", "otsu"],
                        help="Clustering algorithm (default: hdbscan)")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters for kmeans/gmm/agglomerative (default: 5)")
    parser.add_argument("--linkage", default="ward", choices=["ward", "complete", "average", "single"],
                        help="Agglomerative linkage (default: ward)")
    parser.add_argument("--xi", type=float, default=0.05, help="OPTICS xi parameter (default: 0.05)")
    parser.add_argument("--no-snv", action="store_true", help="Skip SNV normalisation (keep absolute spectral levels)")
    parser.add_argument("--pca-components", type=int, default=None, help="Fix number of PCA components (default: auto 95%% variance)")
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine"], help="Distance metric for HDBSCAN (default: euclidean)")
    parser.add_argument("--pixel-mode", action="store_true", help="Cluster on LAB pixel values from REFLECTANCE PNG instead of HSI spectra")
    parser.add_argument("--pixel-weight", type=float, default=0.0, help="Weight of LAB visual features mixed into spectral features (0=pure spectral, e.g. 1.0=equal, 2.0=double visual)")
    parser.add_argument("--spatial-weight", type=float, default=0.0, help="Weight of (y,x) spatial coordinates added to feature vector to enforce spatial coherence (e.g. 0.5, 1.0, 2.0)")
    parser.add_argument("--superpixel", action="store_true", help="Use SLIC superpixel clustering instead of pixel-level clustering")
    parser.add_argument("--n-superpixels", type=int, default=200, help="Target number of SLIC superpixels (default: 200)")
    parser.add_argument("--slic-compactness", type=float, default=5.0, help="SLIC compactness: higher=squarer superpixels, lower=follows edges (default: 5)")
    parser.add_argument("--segment-method", default="slic", choices=["slic", "felzenszwalb", "watershed"],
                        help="Oversegmentation method (default: slic)")
    parser.add_argument("--segment-scale", type=float, default=100.0, help="Felzenszwalb scale parameter, larger=coarser (default: 100)")
    parser.add_argument("--segment-sigma", type=float, default=0.8, help="Felzenszwalb/watershed Gaussian blur sigma (default: 0.8)")
    parser.add_argument("--merge-small", type=float, default=0.0, help="Merge clusters smaller than this fraction of total ROI into nearest large cluster (e.g. 0.05 = merge clusters < 5%%)")
    parser.add_argument("--min-roi-pixels", type=int, default=1000, help="ROIs smaller than this pixel count are output as a single cluster (default: 1000)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Discover artifacts that have a verified mask
    mask_files = [f for f in os.listdir(args.roi) if f.endswith("_mask.png")]
    available = {f[: -len("_mask.png")]: os.path.join(args.roi, f) for f in mask_files}

    if args.artifacts:
        to_process = {name: available[name] for name in args.artifacts if name in available}
        missing = [name for name in args.artifacts if name not in available]
        if missing:
            print(f"Warning: no ROI mask found for: {', '.join(missing)}")
    else:
        to_process = available

    print(f"Processing {len(to_process)} artifact(s)  |  output -> {args.output}\n{'=' * 60}")

    for artifact_name, mask_path in sorted(to_process.items()):
        print(f"\n[{artifact_name}]")

        artifact_folder = os.path.join(args.dataset, artifact_name)
        if not os.path.isdir(artifact_folder):
            print(f"  SKIP — artifact folder not found: {artifact_folder}")
            continue

        hsi_dir = os.path.join(artifact_folder, "HSI")
        capture_path = find_capture_path(hsi_dir)
        if capture_path is None:
            print(f"  SKIP — no HSI capture folder under {hsi_dir}")
            continue
        try:
            raw_file, hdr_file = find_hsi_files(capture_path)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        cube = read_hsi_cube(raw_file, hdr_file)
        if cube is None:
            print(f"  SKIP — failed to read HSI cube")
            continue

        # mask_orig is in viewfinder/REFLECTANCE-PNG coords (for visual crops)
        mask_orig = np.array(Image.open(mask_path).convert("L"))

        # rotate mask into cube coordinate space using the datacube_angle
        angle = get_datacube_angle(artifact_folder)
        mask_cube = rotate_mask_to_cube(mask_orig, angle)
        if angle != 0:
            print(f"  datacube_angle={angle}, mask rotated k={int(((-angle)//90)%4)}")

        # Resize to cube spatial dims if they differ
        h, w = cube.shape[:2]
        if mask_cube.shape != (h, w):
            mask_cube = np.array(Image.fromarray(mask_cube).resize((w, h), Image.NEAREST))

        # reflectance crop uses the original (unrotated) mask on the REFLECTANCE PNG
        refl_path = find_reflectance_png(artifact_folder)
        if refl_path:
            crop_path = os.path.join(args.output, f"{artifact_name}_reflectance_crop.png")
            save_cropped_with_outline(refl_path, mask_orig, crop_path)
            print(f"  Reflectance crop saved")
        else:
            print(f"  Warning: no REFLECTANCE_*.png found, skipping crops")

        output_prefix = os.path.join(args.output, artifact_name)

        # Build LAB image in cube space if needed (pixel-mode or pixel-weight mix)
        lab_cube = None
        if (args.pixel_mode or args.pixel_weight > 0) and refl_path:
            from skimage.color import rgb2lab
            rgb_img = np.array(Image.open(refl_path).convert("RGB"))
            h_c, w_c = cube.shape[:2]
            if rgb_img.shape[:2] != (h_c, w_c):
                rgb_img = np.array(Image.fromarray(rgb_img).resize((w_c, h_c), Image.LANCZOS))
            lab_full = rgb2lab(rgb_img).astype(np.float32)
            # rotate to cube space
            if angle != 0:
                k_rot = int(((-angle) // 90) % 4)
                lab_full = np.rot90(lab_full, k=k_rot).copy()
            lab_cube = lab_full

        # In pixel mode, swap the HSI cube for LAB values from the REFLECTANCE PNG
        cluster_cube = cube
        skip_specular = False
        use_snv = not args.no_snv
        pca_components = args.pca_components
        if args.pixel_mode:
            if not refl_path:
                print(f"  SKIP pixel-mode — no REFLECTANCE_*.png found")
                continue
            from skimage.color import rgb2lab
            rgb_img = np.array(Image.open(refl_path).convert("RGB"))
            h_c, w_c = cube.shape[:2]
            if rgb_img.shape[:2] != (h_c, w_c):
                rgb_img = np.array(Image.fromarray(rgb_img).resize((w_c, h_c), Image.LANCZOS))
            lab_img = rgb2lab(rgb_img).astype(np.float32)  # H x W x 3
            cluster_cube = lab_img
            skip_specular = True
            use_snv = False
            if pca_components is None:
                pca_components = 3

        try:
            if args.superpixel:
                if lab_cube is None:
                    if not refl_path:
                        print(f"  SKIP superpixel — no REFLECTANCE_*.png found")
                        continue
                    from skimage.color import rgb2lab
                    rgb_img = np.array(Image.open(refl_path).convert("RGB"))
                    h_c, w_c = cube.shape[:2]
                    if rgb_img.shape[:2] != (h_c, w_c):
                        rgb_img = np.array(Image.fromarray(rgb_img).resize((w_c, h_c), Image.LANCZOS))
                    lab_full = rgb2lab(rgb_img).astype(np.float32)
                    if angle != 0:
                        k_rot = int(((-angle) // 90) % 4)
                        lab_full = np.rot90(lab_full, k=k_rot).copy()
                    lab_cube = lab_full
                k, n_targets = run_superpixel_clustering(
                    cube, mask_cube, lab_cube, output_prefix,
                    algorithm=args.algorithm,
                    n_superpixels=args.n_superpixels,
                    slic_compactness=args.slic_compactness,
                    min_cluster_size=args.min_cluster_size,
                    min_samples=args.min_samples,
                    cluster_selection_epsilon=args.epsilon,
                    cluster_selection_method=args.method,
                    use_snv=not args.no_snv,
                    n_pca_components=args.pca_components,
                    merge_min_fraction=args.merge_small,
                    segment_method=args.segment_method,
                    segment_scale=args.segment_scale,
                    segment_sigma=args.segment_sigma,
                    min_roi_pixels=args.min_roi_pixels,
                )
            else:
                k, n_targets = run_clustering(
                    cluster_cube, mask_cube, output_prefix,
                    algorithm=args.algorithm,
                    n_clusters=args.n_clusters,
                    min_cluster_size=args.min_cluster_size,
                    min_samples=args.min_samples,
                    cluster_selection_epsilon=args.epsilon,
                    cluster_selection_method=args.method,
                    linkage=args.linkage,
                    xi=args.xi,
                    use_snv=use_snv,
                    n_pca_components=pca_components,
                    metric=args.metric,
                    skip_specular=skip_specular,
                    lab_image=lab_cube if not args.pixel_mode else None,
                    pixel_mix_weight=args.pixel_weight,
                    spatial_weight=args.spatial_weight,
                )
            print(f"  OK — {k} clusters, {n_targets} Raman targets")
            if refl_path:
                overlay_path = os.path.join(args.output, f"{artifact_name}_cluster_overlay.png")
                save_cluster_overlay_crop(
                    refl_path,
                    f"{output_prefix}_clusters.png",
                    mask_orig,
                    overlay_path,
                    cube_angle=angle,
                )
                print(f"  Cluster overlay crop saved")
        except Exception as e:
            print(f"  ERROR — {e}")

    print(f"\n{'=' * 60}\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
