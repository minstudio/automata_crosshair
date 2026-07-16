import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw
import json
from scipy import ndimage
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

EXCLUDED_PREFIXES = ['mire', 'calibration']
EXCLUDED_FOLDERS = ['calibration']

DEFAULT_K = 8
AUTO_K = True
K_RANGE = (6, 12)
MIN_CLUSTER_AREA_PX = 20
# Noise pixels forming a connected region this large get promoted to their own cluster
MIN_NOISE_PROMOTE_PX = 200
# After nearest-centroid fill, merge smallest clusters until at most this many remain
MAX_CLUSTERS = 12   # hard ceiling before silhouette refinement kicks in

# Spatial gaussian sigma applied to the raw cube before pixel extraction.
# Blurs single-pixel noise so small pigment patches form coherent clusters
# instead of being discarded as noise.  1.5–2.0 works well for most acquisitions.
SPATIAL_SMOOTH_SIGMA = 1.5

# Erosion applied to the mask before feeding pixels into HDBSCAN
MASK_EROSION_CLUSTERING = 5
# Erosion applied when choosing crosshair target positions (forces well-interior points)
MASK_EROSION_TARGETS = 25


def find_hsi_cube(folder_path):
    """
    Return (raw_path, hdr_path) for the best available HSI cube, preferring
    the pre-corrected reflectance file over the raw capture.

    Priority:
      1. HSI/raw_data/results/REFLECTANCE_*.dat  (camera-corrected reflectance)
      2. HSI/capture/*.raw  or  HSI/raw_data/capture/*.raw  (raw DN fallback)
    """
    hsi_dir = os.path.join(folder_path, "HSI")
    results_dir = os.path.join(hsi_dir, "raw_data", "results")
    if os.path.isdir(results_dir):
        dats = sorted(f for f in os.listdir(results_dir) if f.startswith("REFLECTANCE_") and f.endswith(".dat"))
        hdrs = sorted(f for f in os.listdir(results_dir) if f.startswith("REFLECTANCE_") and f.endswith(".hdr"))
        if dats and hdrs:
            return os.path.join(results_dir, dats[0]), os.path.join(results_dir, hdrs[0])

    for cp in [os.path.join(hsi_dir, "capture"),
               os.path.join(hsi_dir, "raw_data", "capture")]:
        if not os.path.isdir(cp):
            continue
        files = os.listdir(cp)
        raw = next((f for f in files if f.endswith(".raw") and "DARK" not in f and "WHITE" not in f), None)
        hdr = next((f for f in files if f.endswith(".hdr") and "DARK" not in f and "WHITE" not in f), None)
        if raw and hdr:
            return os.path.join(cp, raw), os.path.join(cp, hdr)
    return None, None


def read_hsi_cube(raw_path, hdr_path):
    try:
        with open(hdr_path, 'r') as f:
            header_str = f.read()
        dims = {}
        for line in header_str.split('\n'):
            if '=' in line:
                key, val = line.split('=', 1)
                dims[key.strip()] = val.strip()
        
        lines = int(dims.get('lines', 0))
        samples = int(dims.get('samples', 0))
        bands = int(dims.get('bands', 0))
        dtype_code = dims.get('data type', '4')
        interleave = dims.get('interleave', 'bsq').lower()
        
        dt_map = {'1': np.uint8, '2': np.int16, '3': np.int32, '4': np.float32, '12': np.uint16}
        dtype = dt_map.get(dtype_code, np.float32)
        
        with open(raw_path, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
            
        if interleave == 'bil':
            cube = data.reshape((lines, bands, samples)).transpose(0, 2, 1)
        elif interleave == 'bip':
            cube = data.reshape((lines, samples, bands))
        else:
            cube = data.reshape((bands, lines, samples)).transpose(1, 2, 0)
            
        return cube
    except Exception as e:
        print(f"    Error reading cube: {e}")
        return None

def normalize_minmax(data):
    d_min = np.min(data)
    d_max = np.max(data)
    if d_max == d_min:
        return np.zeros_like(data, dtype=np.uint8)
    norm = (data - d_min) / (d_max - d_min)
    return (norm * 255).astype(np.uint8)


def apply_snv(spectra):
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    std[std == 0] = 1e-10
    return (spectra - mean) / std


def detect_specular_pixels(spectra, intensity_percentile=98, flatness_threshold=0.02):
    mean_intensity = np.mean(spectra, axis=1)
    high_thresh = np.percentile(mean_intensity, intensity_percentile)
    high_intensity = mean_intensity > high_thresh
    
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = spectra / norms
    variance = np.var(normalized, axis=1)
    flat_spectrum = variance < flatness_threshold
    
    return high_intensity & flat_spectrum


def find_best_point(mask):
    labeled, n_features = ndimage.label(mask)
    if n_features == 0:
        return None
    component_sizes = ndimage.sum(mask, labeled, range(1, n_features + 1))
    largest_idx = np.argmax(component_sizes) + 1
    largest_area = int(component_sizes[largest_idx - 1])
    if largest_area < MIN_CLUSTER_AREA_PX:
        return None
    largest_mask = labeled == largest_idx
    cy, cx = ndimage.center_of_mass(largest_mask)
    cy_int, cx_int = int(round(cy)), int(round(cx))
    if not largest_mask[cy_int, cx_int]:
        ys, xs = np.where(largest_mask)
        dists = (ys - cy) ** 2 + (xs - cx) ** 2
        nearest = np.argmin(dists)
        cy_int, cx_int = ys[nearest], xs[nearest]
    return cy_int, cx_int, largest_area


def draw_crosshair(draw, cx, cy, size=12, color="white", outline_color="black"):
    lw = 2
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            draw.line([(cx - size + dx, cy + dy), (cx + size + dx, cy + dy)], fill=outline_color, width=lw)
            draw.line([(cx + dx, cy - size + dy), (cx + dx, cy + size + dy)], fill=outline_color, width=lw)
    draw.line([(cx - size, cy), (cx + size, cy)], fill=color, width=lw)
    draw.line([(cx, cy - size), (cx, cy + size)], fill=color, width=lw)


def find_optimal_k(features, k_range=(3, 10), method='silhouette'):
    k_min, k_max = k_range
    
    if features.shape[0] < k_max * 10:
        k_max = max(k_min, features.shape[0] // 10)
    
    if k_max <= k_min:
        return k_min
    
    scores = []
    k_values = list(range(k_min, k_max + 1))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(features)
        
        if method == 'silhouette':
            score = silhouette_score(features, labels, sample_size=min(5000, len(features)))
            scores.append(score)
        elif method == 'inertia':
            scores.append(kmeans.inertia_)
    
    if method == 'silhouette':
        best_idx = np.argmax(scores)
    else:
        diffs = np.diff(scores)
        second_diffs = np.diff(diffs)
        if len(second_diffs) > 0:
            best_idx = np.argmax(second_diffs) + 1
        else:
            best_idx = 0
    
    return k_values[best_idx]


def get_cmap(n, name='tab10'):
    if n <= 10:
        return plt.cm.get_cmap('tab10', 10)
    elif n <= 20:
        return plt.cm.get_cmap('tab20', 20)
    else:
        return plt.cm.get_cmap('viridis', n)


import argparse

def process_hdbscan_pca():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=str, required=True, help='Path to artifact folder')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    
    folder_path = args.artifact
    folder_name = os.path.basename(os.path.normpath(folder_path))
    DEBUG_DIR = args.output
    
    folder_lower = folder_name.lower()
    
    if any(folder_lower.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return
        
    if folder_lower in EXCLUDED_FOLDERS:
        print(f"Skipping {folder_name} (excluded)")
        return

    mask_file = f"{folder_name}_mask.png"
    mask_path = os.path.join(DEBUG_DIR, mask_file)
    
    if not os.path.exists(mask_path):
        print(f"Mask path {mask_path} not found")
        return
        
    print(f"Processing {folder_name}...")
    
    mask_img = np.array(Image.open(mask_path).convert('L'))
    mask_binary = mask_img > 127
    
    # Erode mask to avoid edge segmentation artifacts when clustering
    mask_binary = ndimage.binary_erosion(mask_binary, iterations=MASK_EROSION_CLUSTERING)

    # Deeper erosion used only for crosshair placement (keeps targets well away from edges)
    mask_inner = ndimage.binary_erosion(mask_binary, iterations=MASK_EROSION_TARGETS - MASK_EROSION_CLUSTERING)
    
    if not np.any(mask_binary):
        print("  Mask is empty, skipping")
        return
        
    if not os.path.exists(folder_path):
        print(f"  Artifact folder not found: {folder_path}")
        return

    raw_file, hdr_file = find_hsi_cube(folder_path)
    if raw_file is None:
        print(f"  No HSI cube found")
        return

    cube = read_hsi_cube(raw_file, hdr_file)
    if cube is None:
        return

    h, w, b = cube.shape

    # Spatially smooth the cube band-by-band before extracting pixels.
    # This blurs single-pixel noise so small pigment patches form coherent
    # density peaks in feature space and are no longer dropped as HDBSCAN noise.
    if SPATIAL_SMOOTH_SIGMA > 0:
        print(f"  Applying spatial smoothing (sigma={SPATIAL_SMOOTH_SIGMA})...")
        cube_f = cube.astype(np.float32)
        for band_idx in range(b):
            cube_f[:, :, band_idx] = ndimage.gaussian_filter(
                cube_f[:, :, band_idx], sigma=SPATIAL_SMOOTH_SIGMA
            )
    else:
        cube_f = cube.astype(np.float32)

    masked_pixels = cube_f[mask_binary]
    
    if masked_pixels.shape[0] < 100:
        print(f"  Too few pixels ({masked_pixels.shape[0]}), skipping")
        return
    
    print("  Detecting specular pixels...")
    specular_mask_local = detect_specular_pixels(masked_pixels)
    n_specular = np.sum(specular_mask_local)
    print(f"  Found {n_specular} specular pixels within mask ({100*n_specular/len(masked_pixels):.1f}%)")
    
    valid_pixels = masked_pixels[~specular_mask_local]
    
    if valid_pixels.shape[0] < 100:
        print(f"  Too few valid pixels after specular removal, using all")
        valid_pixels = masked_pixels
        specular_mask_local = np.zeros(len(masked_pixels), dtype=bool)
    
    print("  Applying SNV normalization...")
    pixels_snv = apply_snv(valid_pixels)
    
    print("  Computing PCA (SNV features)...")
    n_components = min(25, valid_pixels.shape[1], valid_pixels.shape[0] // 10)
    pca = PCA(n_components=n_components)
    pca_features_snv = pca.fit_transform(pixels_snv)

    # Trim to 95% explained variance (avoids fitting noise dimensions on uniform artifacts)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = max(3, int(np.searchsorted(cumvar, 0.95)) + 1)
    n_keep = min(n_keep, n_components)
    pca_features_snv = pca_features_snv[:, :n_keep]
    print(f"  Kept {n_keep}/{n_components} PCA components ({cumvar[n_keep-1]*100:.1f}% variance)")

    # Standardise so all components contribute equally to Euclidean distance.
    # Without this, high-variance PCs dominate and distance scale varies per artifact.
    pca_features = StandardScaler().fit_transform(pca_features_snv)
    print(f"  Using {pca_features.shape[1]} features ({n_keep} SNV-PCA standardised)")

    n_valid = valid_pixels.shape[0]
    # min_cluster_size: with 'leaf' selection we find many small clusters, so we
    # use a larger floor (30) to avoid individual-noise clusters while still
    # catching small pigment patches.  Hard cap at 80 for very large acquisitions.
    min_cluster_size = max(30, min(80, n_valid // 150))
    min_samples      = max(3,  min(10, n_valid // 800))
    # epsilon=0: do not force-merge sub-clusters.  With 'leaf' selection this lets
    # spectral sub-groups that EOM would have merged remain as separate clusters.
    cse = 0.0

    print(f"  Computing HDBSCAN (mcs={min_cluster_size}, ms={min_samples}, epsilon={cse})...")
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        # 'leaf' selects the bottom-of-tree clusters rather than one large merged
        # blob, which is critical for separating distinct pigment/material regions.
        cluster_selection_method='leaf',
        cluster_selection_epsilon=cse,
        allow_single_cluster=True,
    )
    valid_labels = hdbscan.fit_predict(pca_features)

    unique_labels = set(valid_labels)
    k = len([l for l in unique_labels if l >= 0])

    # --- Fallback: if no clusters found, relax parameters progressively ---
    if k == 0:
        fallback_params = [
            dict(min_cluster_size=max(15, n_valid // 250), min_samples=max(3, n_valid // 1200),
                 cluster_selection_method='leaf', cluster_selection_epsilon=0.0,
                 allow_single_cluster=True),
            dict(min_cluster_size=10, min_samples=2,
                 cluster_selection_method='leaf', cluster_selection_epsilon=0.0,
                 allow_single_cluster=True),
            dict(min_cluster_size=5, min_samples=1,
                 cluster_selection_method='leaf', cluster_selection_epsilon=0.0,
                 allow_single_cluster=True),
        ]
        for attempt, params in enumerate(fallback_params, start=1):
            print(f"  [!] No clusters found. Retrying with more forgiving parameters "
                  f"(attempt {attempt}/{len(fallback_params)}): {params}")
            hdbscan_retry = HDBSCAN(metric='euclidean', **params)
            valid_labels = hdbscan_retry.fit_predict(pca_features)
            unique_labels = set(valid_labels)
            k = len([l for l in unique_labels if l >= 0])
            if k > 0:
                print(f"  [OK] Found {k} cluster(s) on retry attempt {attempt}.")
                break
        if k == 0:
            print(f"  [!] Could not find any clusters even with most forgiving parameters. "
                  f"All pixels will be treated as noise.")
    
    full_labels = np.full(len(masked_pixels), -2, dtype=int)
    full_labels[~specular_mask_local] = valid_labels

    # ── Post-processing: discard edge clusters & keep largest components ──
    # Build a spatial label image so we can reason about geometry
    label_img = np.full((h, w), -2, dtype=int)
    label_img[mask_binary] = full_labels

    # Edge proximity: pixels within MASK_EROSION_CLUSTERING+5 of the mask boundary
    edge_band = MASK_EROSION_CLUSTERING + 5
    mask_eroded_edge = ndimage.binary_erosion(mask_binary, iterations=edge_band)
    near_edge = mask_binary & ~mask_eroded_edge  # ring of pixels near mask boundary

    # Mask centroid (used to rank clusters by how "central" they are)
    ys_mask, xs_mask = np.where(mask_binary)
    mask_cy, mask_cx = ys_mask.mean(), xs_mask.mean()

    cluster_ids = sorted([l for l in set(full_labels) if l >= 0])
    keep_ids = []
    for cid in cluster_ids:
        cluster_pixels = (label_img == cid)
        total = np.sum(cluster_pixels)
        if total < MIN_CLUSTER_AREA_PX:
            continue
        edge_frac = np.sum(cluster_pixels & near_edge) / max(total, 1)
        if edge_frac > 0.50:
            print(f"  Discarding cluster {cid}: {edge_frac:.0%} of pixels on mask edge (background bleed)")
            label_img[cluster_pixels] = -1  # demote to noise
            continue
        # Keep only the largest connected component of this cluster
        labeled_cc, n_cc = ndimage.label(cluster_pixels)
        if n_cc > 1:
            cc_sizes = ndimage.sum(cluster_pixels, labeled_cc, range(1, n_cc + 1))
            largest_cc = np.argmax(cc_sizes) + 1
            small_cc = cluster_pixels & (labeled_cc != largest_cc)
            removed_px = int(np.sum(small_cc))
            if removed_px > 0:
                label_img[small_cc] = -1
                print(f"  Cluster {cid}: kept largest component, removed {removed_px} scattered px")
        keep_ids.append(cid)

    # Nearest-centroid assignment: assign every unclustered / specular mask pixel
    # to the closest cluster centroid in PCA feature space.  This avoids the old
    # behaviour where 90%+ of the artifact was lumped into a single gray
    # "promoted noise" cluster, producing a colourless visualization.
    if keep_ids:
        # Build lookup: spatial (y,x) → index in masked_pixels / full_labels
        ys_mask_flat, xs_mask_flat = np.where(mask_binary)
        n_masked_flat = len(ys_mask_flat)
        mask_idx_2d = np.full((h, w), -1, dtype=np.int32)
        mask_idx_2d[ys_mask_flat, xs_mask_flat] = np.arange(n_masked_flat)

        # Build lookup: masked_pixel_index → pca_features row (-1 if specular)
        pca_row = np.full(n_masked_flat, -1, dtype=np.int32)
        non_spec_idx = np.where(~specular_mask_local)[0]
        pca_row[non_spec_idx] = np.arange(len(non_spec_idx))

        # Compute centroid for each kept cluster from the post-processed label_img
        centroid_list, centroid_ids = [], []
        for cid in keep_ids:
            ys_c, xs_c = np.where(label_img == cid)
            if len(ys_c) == 0:
                continue
            vi = pca_row[mask_idx_2d[ys_c, xs_c]]
            vi_valid = vi[vi >= 0]
            if len(vi_valid) == 0:
                continue
            centroid_list.append(pca_features[vi_valid].mean(axis=0))
            centroid_ids.append(cid)

        if centroid_ids:
            centroid_arr = np.stack(centroid_list)  # (n_clusters, n_features)

            # All pixels inside the mask that have no valid cluster assignment
            ys_u, xs_u = np.where(mask_binary & ~np.isin(label_img, keep_ids))
            if len(ys_u):
                mi_u  = mask_idx_2d[ys_u, xs_u]
                vi_u  = pca_row[mi_u]
                has_pca = vi_u >= 0

                # Non-specular: assign via nearest centroid in PCA space
                if np.any(has_pca):
                    feats = pca_features[vi_u[has_pca]]
                    dists = np.linalg.norm(feats[:, None] - centroid_arr[None], axis=2)
                    label_img[ys_u[has_pca], xs_u[has_pca]] = \
                        np.array(centroid_ids)[np.argmin(dists, axis=1)]

                # Specular: no PCA features — assign to nearest cluster by raw mean intensity
                if np.any(~has_pca):
                    raw_spec = masked_pixels[mi_u[~has_pca]].astype(np.float32)
                    raw_means = raw_spec.mean(axis=1)
                    # compute cluster mean raw intensity as a simple proxy
                    clust_raw_means = []
                    for cid in centroid_ids:
                        ys_c2, xs_c2 = np.where(label_img == cid)
                        if len(ys_c2):
                            mi_c2 = mask_idx_2d[ys_c2, xs_c2]
                            clust_raw_means.append(masked_pixels[mi_c2].mean())
                        else:
                            clust_raw_means.append(0.0)
                    clust_raw_means = np.array(clust_raw_means)
                    nearest_clust = np.array(centroid_ids)[
                        np.argmin(np.abs(raw_means[:, None] - clust_raw_means[None]), axis=1)
                    ]
                    label_img[ys_u[~has_pca], xs_u[~has_pca]] = nearest_clust

            # keep_ids may now include cluster ids that gained pixels from the fill
            keep_ids = sorted({int(v) for v in label_img[mask_binary] if v >= 0})
            print(f"  Nearest-centroid fill: {int(np.sum(mask_binary))} mask px -> {len(keep_ids)} clusters")
    else:
        # No clusters at all – treat entire mask as one region
        label_img[mask_binary] = 0
        keep_ids = [0]

    # Hard ceiling pass (fast, before the silhouette search)
    while len(keep_ids) > MAX_CLUSTERS:
        cents, sizes = {}, {}
        for cid in keep_ids:
            ys_c2, xs_c2 = np.where(label_img == cid)
            sizes[cid] = len(ys_c2)
            mi2 = mask_idx_2d[ys_c2, xs_c2]
            vi2 = pca_row[mi2]; vi2_v = vi2[vi2 >= 0]
            if len(vi2_v):
                cents[cid] = pca_features[vi2_v].mean(axis=0)
        valid_cids = [c for c in keep_ids if c in cents]
        if len(valid_cids) <= 1:
            break
        tiny = min(valid_cids, key=lambda c: sizes[c])
        others = [c for c in valid_cids if c != tiny]
        absorb = others[int(np.argmin([np.linalg.norm(cents[tiny] - cents[o]) for o in others]))]
        label_img[label_img == tiny] = absorb
        keep_ids = [c for c in keep_ids if c != tiny]
    keep_ids = sorted({int(v) for v in label_img[mask_binary] if v >= 0})

    # Silhouette-guided merge: build a full agglomerative hierarchy from the
    # current clusters (greedy nearest-centroid merges) and pick the k that
    # maximises silhouette score on pca_features.  This selects the "natural"
    # number of clusters without any hard-coded target.
    if len(keep_ids) >= 3:
        # Label for each pca_features row (non-specular masked pixels, in order)
        pca_labels_cur = label_img[ys_mask_flat[non_spec_idx], xs_mask_flat[non_spec_idx]]

        # Build hierarchy: snapshot label arrays at each k level (k..2)
        snap_labels = [pca_labels_cur.copy()]
        snap_limg   = [label_img.copy()]
        w_labels    = pca_labels_cur.copy()
        w_img       = label_img.copy()
        w_ids       = keep_ids[:]

        while len(w_ids) > 2:
            cents = {}
            for cid in w_ids:
                m = w_labels == cid
                if m.any():
                    cents[cid] = pca_features[m].mean(axis=0)
            ids_c = list(cents.keys())
            if len(ids_c) < 2:
                break
            best_d, mfrom, minto = np.inf, None, None
            for i in range(len(ids_c)):
                for j in range(i + 1, len(ids_c)):
                    d = np.linalg.norm(cents[ids_c[i]] - cents[ids_c[j]])
                    if d < best_d:
                        best_d, mfrom, minto = d, ids_c[i], ids_c[j]
            w_labels[w_labels == mfrom] = minto
            w_img[w_img == mfrom] = minto
            w_ids = [c for c in w_ids if c != mfrom]
            snap_labels.append(w_labels.copy())
            snap_limg.append(w_img.copy())

        # Evaluate silhouette at each snapshot; pick best k
        sample_n = min(5000, len(pca_features))
        best_score, best_idx = -np.inf, 0
        for idx, sl in enumerate(snap_labels):
            if len(np.unique(sl)) < 2:
                continue
            try:
                sc = silhouette_score(pca_features, sl, sample_size=sample_n, random_state=42)
            except Exception:
                sc = -1.0
            if sc > best_score:
                best_score, best_idx = sc, idx

        label_img = snap_limg[best_idx]
        keep_ids  = sorted({int(v) for v in label_img[mask_binary] if v >= 0})
        print(f"  Silhouette auto-k={len(keep_ids)} (score={best_score:.3f})")

    keep_ids = sorted({int(v) for v in label_img[mask_binary] if v >= 0})

    # Re-map to contiguous 0..k-1
    remap = {old: new for new, old in enumerate(keep_ids)}
    k = len(keep_ids)
    print(f"  After filtering: {k} clusters kept")

    cmap = get_cmap(max(k, 1))
    label_colors = (cmap(np.arange(max(k, 1)))[:, :3] * 255).astype(np.uint8)

    out_img = np.zeros((h, w, 3), dtype=np.uint8)
    for old_id, new_id in remap.items():
        out_img[label_img == old_id] = label_colors[new_id]
    out_img[(label_img == -1) & mask_binary] = [40, 40, 40]
    
    # Identify Raman points and draw crosshairs
    vis = Image.fromarray(out_img)
    draw = ImageDraw.Draw(vis)
    
    targets = []
    for i in range(k):
        color_rgb = label_colors[i]
        cluster_mask = np.all(out_img == color_rgb, axis=2)
        # Prefer targets strictly inside the deep-eroded inner region;
        # fall back to the full cluster mask if the inner region has no pixels for this cluster.
        cluster_inner = cluster_mask & mask_inner
        result = find_best_point(cluster_inner) or find_best_point(cluster_mask)
        if result:
            row, col, area = result
            draw_crosshair(draw, col, row, size=10)
            targets.append({"cluster": int(i), "x": int(col), "y": int(row)})
            # Draw label
            draw.text((col + 5, row - 10), str(i), fill="white")
    
    out_path = os.path.join(DEBUG_DIR, f"{folder_name}_hdbscan_pca.png")
    vis.save(out_path)
    
    print(f"  Saved HDBSCAN visualization with crosshairs ({k} clusters, {len(targets)} targets)")
    
    # Save targets
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    json_path = os.path.join(DEBUG_DIR, f"{folder_name}_raman_targets.json")
    with open(json_path, "w") as f:
        json.dump({"artifact": folder_name, "targets": targets}, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    process_hdbscan_pca()
