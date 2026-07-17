import os
import json
import argparse
import warnings
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MIN_MASK_AREA_PX = 500
MASK_CLOSE_ITERS = 4
MASK_OPEN_ITERS = 2
MASK_FILL_HOLES = True

MASK_EROSION_CLUSTERING = 5
MASK_EROSION_TARGETS = 25

MIN_CLUSTER_AREA_PX = 20
MIN_NOISE_PROMOTE_PX = 200


def read_hsi_cube(raw_path, hdr_path):
    with open(hdr_path, "r") as f:
        header_str = f.read()
    dims = {}
    for line in header_str.split("\n"):
        if "=" in line:
            key, val = line.split("=", 1)
            dims[key.strip().lower()] = val.strip()
    lines = int(dims.get("lines", 0))
    samples = int(dims.get("samples", 0))
    bands = int(dims.get("bands", 0))
    dtype_code = dims.get("data type", "4")
    interleave = dims.get("interleave", "bsq").lower()
    dt_map = {"1": np.uint8, "2": np.int16, "3": np.int32, "4": np.float32, "12": np.uint16}
    dtype = dt_map.get(dtype_code, np.float32)
    with open(raw_path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    if interleave == "bil":
        cube = data.reshape((lines, bands, samples)).transpose(0, 2, 1)
    elif interleave == "bip":
        cube = data.reshape((lines, samples, bands))
    else:
        cube = data.reshape((bands, lines, samples)).transpose(1, 2, 0)
    return cube


def normalize_minmax(data):
    d_min = np.nanmin(data)
    d_max = np.nanmax(data)
    if d_max == d_min:
        return np.zeros_like(data, dtype=np.uint8)
    norm = (data - d_min) / (d_max - d_min)
    return (np.clip(norm, 0, 1) * 255).astype(np.uint8)


def apply_snv(spectra):
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    std[std == 0] = 1e-10
    return (spectra - mean) / std


def find_capture_path(hsi_dir):
    direct = os.path.join(hsi_dir, "capture")
    if os.path.isdir(direct):
        return direct
    nested = os.path.join(hsi_dir, "raw_data", "capture")
    if os.path.isdir(nested):
        return nested
    return None


def find_hsi_files(capture_path):
    files = os.listdir(capture_path)
    raw_file = next((f for f in files if f.endswith(".raw") and "DARK" not in f and "WHITE" not in f), None)
    hdr_file = next((f for f in files if f.endswith(".hdr") and "DARK" not in f and "WHITE" not in f), None)
    if not raw_file or not hdr_file:
        raise FileNotFoundError("Missing .raw/.hdr (excluding DARK/WHITE)")
    return os.path.join(capture_path, raw_file), os.path.join(capture_path, hdr_file)


def largest_component(binary):
    labeled, n = ndimage.label(binary)
    if n == 0:
        return binary * 0
    areas = ndimage.sum(binary, labeled, range(1, n + 1))
    best = int(np.argmax(areas) + 1)
    return labeled == best


def cleanup_mask(mask):
    mask = mask.astype(bool)
    if np.sum(mask) < MIN_MASK_AREA_PX:
        return mask.astype(np.uint8) * 255
    mask = ndimage.binary_closing(mask, iterations=MASK_CLOSE_ITERS)
    mask = ndimage.binary_opening(mask, iterations=MASK_OPEN_ITERS)
    if MASK_FILL_HOLES:
        mask = ndimage.binary_fill_holes(mask)
    mask = largest_component(mask)
    return mask.astype(np.uint8) * 255


def score_cluster_as_artifact(labels_img, cid):
    mask = labels_img == cid
    area = float(np.sum(mask))
    if area <= 0:
        return -1.0
    h, w = labels_img.shape
    cy, cx = ndimage.center_of_mass(mask)
    img_cy, img_cx = h / 2.0, w / 2.0
    dist = float(np.sqrt((cy - img_cy) ** 2 + (cx - img_cx) ** 2))
    max_dist = float(np.sqrt(img_cy**2 + img_cx**2) + 1e-10)
    centrality = 1.0 - (dist / max_dist)
    border = 2
    touches_border = (
        np.any(mask[:border, :])
        or np.any(mask[-border:, :])
        or np.any(mask[:, :border])
        or np.any(mask[:, -border:])
    )
    penalty = 0.35 if touches_border else 0.0
    return area * (0.25 + 0.75 * centrality) * (1.0 - penalty)


def auto_mask_from_pca_kmeans(cube, n_components=10, k=5, sample_cap=120000, random_state=42):
    h, w, b = cube.shape
    vectors = cube.reshape(-1, b).astype(np.float32)
    keep = np.isfinite(vectors).all(axis=1)
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    if len(idx) > sample_cap:
        rng = np.random.default_rng(random_state)
        idx_s = rng.choice(idx, size=sample_cap, replace=False)
    else:
        idx_s = idx
    snv = apply_snv(vectors[idx_s])
    n_components = int(min(n_components, b, max(3, len(idx_s) // 20)))
    pca = PCA(n_components=n_components, random_state=random_state)
    feats = pca.fit_transform(snv)
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)
    km = MiniBatchKMeans(n_clusters=int(k), random_state=random_state, n_init=3, batch_size=4096)
    km.fit(feats)
    snv_all = apply_snv(vectors[idx])
    feats_all = pca.transform(snv_all)
    feats_all = scaler.transform(feats_all)
    labels = km.predict(feats_all)
    labels_img = np.full(h * w, -1, dtype=np.int32)
    labels_img[idx] = labels
    labels_img = labels_img.reshape(h, w)
    best_cid = None
    best_score = -1.0
    for cid in range(int(k)):
        s = score_cluster_as_artifact(labels_img, cid)
        if s > best_score:
            best_score = s
            best_cid = cid
    if best_cid is None:
        return np.zeros((h, w), dtype=np.uint8)
    mask = labels_img == best_cid
    return cleanup_mask(mask)


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


def get_cmap(n):
    if n <= 10:
        return plt.cm.get_cmap("tab10", 10)
    if n <= 20:
        return plt.cm.get_cmap("tab20", 20)
    return plt.cm.get_cmap("viridis", n)


def find_best_point(mask):
    labeled, n_features = ndimage.label(mask)
    if n_features == 0:
        return None
    component_sizes = ndimage.sum(mask, labeled, range(1, n_features + 1))
    largest_idx = int(np.argmax(component_sizes) + 1)
    largest_area = int(component_sizes[largest_idx - 1])
    if largest_area < MIN_CLUSTER_AREA_PX:
        return None
    largest_mask = labeled == largest_idx
    cy, cx = ndimage.center_of_mass(largest_mask)
    cy_int, cx_int = int(round(cy)), int(round(cx))
    if cy_int < 0 or cy_int >= largest_mask.shape[0] or cx_int < 0 or cx_int >= largest_mask.shape[1]:
        return None
    if not largest_mask[cy_int, cx_int]:
        ys, xs = np.where(largest_mask)
        if len(ys) == 0:
            return None
        dists = (ys - cy) ** 2 + (xs - cx) ** 2
        nearest = int(np.argmin(dists))
        cy_int, cx_int = int(ys[nearest]), int(xs[nearest])
    return cy_int, cx_int, largest_area


def draw_crosshair(draw, cx, cy, size=12, color="white", outline_color="black"):
    lw = 2
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            draw.line([(cx - size + dx, cy + dy), (cx + size + dx, cy + dy)], fill=outline_color, width=lw)
            draw.line([(cx + dx, cy - size + dy), (cx + dx, cy + size + dy)], fill=outline_color, width=lw)
    draw.line([(cx - size, cy), (cx + size, cy)], fill=color, width=lw)
    draw.line([(cx, cy - size), (cx, cy + size)], fill=color, width=lw)


def run_hdbscan_clustering(cube, mask_u8, output_prefix,
                            min_cluster_size=None, min_samples=None,
                            cluster_selection_epsilon=0.3,
                            cluster_selection_method="eom"):
    try:
        from hdbscan import HDBSCAN
    except Exception:
        from sklearn.cluster import HDBSCAN

    mask_binary = mask_u8 > 127
    mask_binary = ndimage.binary_erosion(mask_binary, iterations=MASK_EROSION_CLUSTERING)
    mask_inner = ndimage.binary_erosion(mask_binary, iterations=max(0, MASK_EROSION_TARGETS - MASK_EROSION_CLUSTERING))
    if not np.any(mask_binary):
        raise ValueError("Mask empty after erosion")

    h, w, b = cube.shape
    masked_pixels = cube[mask_binary].astype(np.float32)
    if masked_pixels.shape[0] < 100:
        raise ValueError(f"Too few masked pixels ({masked_pixels.shape[0]})")

    specular_local = detect_specular_pixels(masked_pixels)
    valid_pixels = masked_pixels[~specular_local]
    if valid_pixels.shape[0] < 100:
        valid_pixels = masked_pixels
        specular_local = np.zeros(len(masked_pixels), dtype=bool)

    pixels_snv = apply_snv(valid_pixels)
    n_components = int(min(25, valid_pixels.shape[1], max(3, valid_pixels.shape[0] // 10)))
    pca = PCA(n_components=n_components)
    pca_features_snv = pca.fit_transform(pixels_snv)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(max(3, int(np.searchsorted(cumvar, 0.95)) + 1))
    n_keep = int(min(n_keep, n_components))
    pca_features_snv = pca_features_snv[:, :n_keep]
    pca_features_snv = StandardScaler().fit_transform(pca_features_snv)
    mean_intensity = np.mean(valid_pixels, axis=1, keepdims=True)
    mean_intensity = (mean_intensity - mean_intensity.mean()) / (mean_intensity.std() + 1e-10)
    pca_features = np.hstack([pca_features_snv, mean_intensity * 0.25])

    n_valid = valid_pixels.shape[0]
    if min_cluster_size is None:
        min_cluster_size = int(max(10, min(30, n_valid // 400)))
    if min_samples is None:
        min_samples = int(max(3, min(8, n_valid // 1000)))
    hdbscan = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        allow_single_cluster=True,
    )
    try:
        valid_labels = hdbscan.fit_predict(pca_features)
    except TypeError:
        # sklearn bug: epsilon > 0 crashes on certain data distributions
        hdbscan = HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            metric="euclidean",
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=0.0,
            allow_single_cluster=True,
        )
        valid_labels = hdbscan.fit_predict(pca_features)
    k = int(len([l for l in set(valid_labels) if l >= 0]))

    if k == 0:
        fallback_params = [
            dict(
                min_cluster_size=int(max(8, n_valid // 600)),
                min_samples=int(max(2, n_valid // 1500)),
                cluster_selection_method="eom",
                cluster_selection_epsilon=0.3,
                allow_single_cluster=True,
            ),
            dict(
                min_cluster_size=5,
                min_samples=2,
                cluster_selection_method="eom",
                cluster_selection_epsilon=0.0,
                allow_single_cluster=True,
            ),
            dict(
                min_cluster_size=3,
                min_samples=1,
                cluster_selection_method="leaf",
                cluster_selection_epsilon=0.0,
                allow_single_cluster=True,
            ),
        ]
        for params in fallback_params:
            hdbscan_retry = HDBSCAN(metric="euclidean", **params)
            try:
                valid_labels = hdbscan_retry.fit_predict(pca_features)
            except TypeError:
                params["cluster_selection_epsilon"] = 0.0
                hdbscan_retry = HDBSCAN(metric="euclidean", **params)
                valid_labels = hdbscan_retry.fit_predict(pca_features)
            k = int(len([l for l in set(valid_labels) if l >= 0]))
            if k > 0:
                break

    full_labels = np.full(len(masked_pixels), -2, dtype=int)
    full_labels[~specular_local] = valid_labels

    label_img = np.full((h, w), -2, dtype=int)
    label_img[mask_binary] = full_labels

    edge_band = int(MASK_EROSION_CLUSTERING + 5)
    mask_eroded_edge = ndimage.binary_erosion(mask_binary, iterations=edge_band)
    near_edge = mask_binary & ~mask_eroded_edge

    keep_ids = []
    cluster_ids = sorted([l for l in set(full_labels) if l >= 0])
    for cid in cluster_ids:
        cluster_pixels = label_img == cid
        total = int(np.sum(cluster_pixels))
        if total < MIN_CLUSTER_AREA_PX:
            continue
        edge_frac = float(np.sum(cluster_pixels & near_edge) / max(total, 1))
        if edge_frac > 0.50:
            label_img[cluster_pixels] = -1
            continue
        keep_ids.append(cid)

    unclustered = mask_binary & ~np.isin(label_img, keep_ids)
    unclust_count = int(np.sum(unclustered))
    if unclust_count >= MIN_NOISE_PROMOTE_PX:
        next_id = int((max(keep_ids) + 1) if keep_ids else 0)
        label_img[unclustered] = next_id
        keep_ids.append(next_id)

    keep_ids = list(dict.fromkeys(keep_ids))
    remap = {old: new for new, old in enumerate(keep_ids)}
    k = int(len(keep_ids))

    cmap = get_cmap(max(k, 1))
    label_colors = (cmap(np.arange(max(k, 1)))[:, :3] * 255).astype(np.uint8)

    out_img = np.zeros((h, w, 3), dtype=np.uint8)
    for old_id, new_id in remap.items():
        out_img[label_img == old_id] = label_colors[new_id]
    out_img[(label_img == -1) & mask_binary] = [40, 40, 40]

    vis = Image.fromarray(out_img)
    draw = ImageDraw.Draw(vis)
    targets = []
    for i in range(k):
        color_rgb = label_colors[i]
        cluster_mask = np.all(out_img == color_rgb, axis=2)
        cluster_inner = cluster_mask & mask_inner
        result = find_best_point(cluster_inner) or find_best_point(cluster_mask)
        if result:
            row, col, _area = result
            draw_crosshair(draw, col, row, size=10)
            draw.text((col + 5, row - 10), str(i), fill="white")
            targets.append({"cluster": int(i), "x": int(col), "y": int(row)})

    Image.fromarray(mask_u8).save(f"{output_prefix}_mask.png")
    Image.fromarray(out_img).save(f"{output_prefix}_hdbscan_pca.png")
    with open(f"{output_prefix}_raman_targets.json", "w") as f:
        json.dump({"targets": targets}, f, indent=2)
    return k, len(targets)


def run_clustering(cube, mask_u8, output_prefix,
                   algorithm="hdbscan",
                   n_clusters=5,
                   min_cluster_size=None, min_samples=None,
                   cluster_selection_epsilon=0.3,
                   cluster_selection_method="eom",
                   linkage="ward",
                   xi=0.05,
                   use_snv=True,
                   n_pca_components=None,
                   metric="euclidean",
                   skip_specular=False,
                   lab_image=None,
                   pixel_mix_weight=0.0,
                   spatial_weight=0.0):
    """
    General clustering entry point. Supports: hdbscan, kmeans, gmm, agglomerative, optics.
    Saves {output_prefix}_clusters.png, _mask.png, _raman_targets.json.
    Returns (k, n_targets).
    """
    mask_binary = mask_u8 > 127
    mask_binary = ndimage.binary_erosion(mask_binary, iterations=MASK_EROSION_CLUSTERING)
    mask_inner = ndimage.binary_erosion(mask_binary, iterations=max(0, MASK_EROSION_TARGETS - MASK_EROSION_CLUSTERING))
    if not np.any(mask_binary):
        raise ValueError("Mask empty after erosion")

    h, w, b = cube.shape
    masked_pixels = cube[mask_binary].astype(np.float32)
    if masked_pixels.shape[0] < 100:
        raise ValueError(f"Too few masked pixels ({masked_pixels.shape[0]})")

    if skip_specular:
        specular_local = np.zeros(len(masked_pixels), dtype=bool)
        valid_pixels = masked_pixels
    else:
        specular_local = detect_specular_pixels(masked_pixels)
        valid_pixels = masked_pixels[~specular_local]
        if valid_pixels.shape[0] < 100:
            valid_pixels = masked_pixels
            specular_local = np.zeros(len(masked_pixels), dtype=bool)

    spectra = apply_snv(valid_pixels) if use_snv else valid_pixels.copy()
    if n_pca_components is not None:
        n_components = int(min(n_pca_components, spectra.shape[1], spectra.shape[0] - 1))
        n_components = max(1, n_components)
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(spectra)
    else:
        n_components = int(min(25, spectra.shape[1], max(3, spectra.shape[0] // 10)))
        pca = PCA(n_components=n_components)
        pca_features_all = pca.fit_transform(spectra)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(max(3, int(np.searchsorted(cumvar, 0.95)) + 1))
        n_keep = int(min(n_keep, n_components))
        pca_features = pca_features_all[:, :n_keep]
    pca_features = StandardScaler().fit_transform(pca_features)
    if use_snv:
        mean_intensity = np.mean(valid_pixels, axis=1, keepdims=True)
        mean_intensity = (mean_intensity - mean_intensity.mean()) / (mean_intensity.std() + 1e-10)
        pca_features = np.hstack([pca_features, mean_intensity * 0.25])

    # Optionally mix in LAB visual features from the reflectance image
    if lab_image is not None and pixel_mix_weight > 0:
        h_l, w_l = lab_image.shape[:2]
        if lab_image.shape[:2] != (h, w):
            lab_image = np.array(Image.fromarray(
                lab_image.astype(np.float32) if lab_image.dtype != np.uint8 else lab_image
            ).resize((w, h), Image.LANCZOS))
        lab_pixels = lab_image[mask_binary].astype(np.float32)
        lab_valid = lab_pixels[~specular_local]
        lab_scaled = StandardScaler().fit_transform(lab_valid)
        pca_features = np.hstack([pca_features, lab_scaled * pixel_mix_weight])

    # Optionally mix in (y, x) spatial coordinates to enforce spatial coherence
    if spatial_weight > 0:
        ys, xs = np.where(mask_binary)
        coords = np.column_stack([ys, xs]).astype(np.float32)
        coords_scaled = StandardScaler().fit_transform(coords)
        coords_valid = coords_scaled[~specular_local]
        pca_features = np.hstack([pca_features, coords_valid * spatial_weight])

    n_valid = valid_pixels.shape[0]
    algo = algorithm.lower()

    if algo == "hdbscan":
        try:
            from hdbscan import HDBSCAN
        except Exception:
            from sklearn.cluster import HDBSCAN
        mcs = int(min_cluster_size) if min_cluster_size is not None else int(max(10, min(30, n_valid // 400)))
        ms = int(min_samples) if min_samples is not None else int(max(3, min(8, n_valid // 1000)))
        try:
            valid_labels = HDBSCAN(
                min_cluster_size=mcs, min_samples=ms, metric=metric,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=cluster_selection_epsilon,
                allow_single_cluster=True,
            ).fit_predict(pca_features)
        except TypeError:
            valid_labels = HDBSCAN(
                min_cluster_size=mcs, min_samples=ms, metric=metric,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=0.0,
                allow_single_cluster=True,
            ).fit_predict(pca_features)

    elif algo == "kmeans":
        from sklearn.cluster import KMeans
        valid_labels = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=42).fit_predict(pca_features)

    elif algo == "gmm":
        from sklearn.mixture import GaussianMixture
        valid_labels = GaussianMixture(n_components=int(n_clusters), random_state=42, n_init=3).fit_predict(pca_features)

    elif algo == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        valid_labels = AgglomerativeClustering(n_clusters=int(n_clusters), linkage=linkage).fit_predict(pca_features)

    elif algo == "optics":
        from sklearn.cluster import OPTICS
        ms_optics = int(min_samples) if min_samples is not None else int(max(3, min(8, n_valid // 1000)))
        valid_labels = OPTICS(min_samples=ms_optics, xi=xi, metric="euclidean").fit_predict(pca_features)

    elif algo == "meanshift":
        from sklearn.cluster import MeanShift, estimate_bandwidth
        bw = estimate_bandwidth(pca_features, quantile=0.2, n_samples=min(5000, len(pca_features)))
        if bw <= 0:
            bw = 0.5
        valid_labels = MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(pca_features)

    elif algo == "bgmm":
        from sklearn.mixture import BayesianGaussianMixture
        max_k = int(n_clusters)
        bgmm = BayesianGaussianMixture(n_components=max_k, random_state=42, n_init=3,
                                        weight_concentration_prior=1e-2)
        bgmm.fit(pca_features)
        valid_labels = bgmm.predict(pca_features)
        # remap to contiguous ids (prunes near-zero weight components)
        active = np.where(bgmm.weights_ > 0.01)[0]
        remap_bgmm = {old: new for new, old in enumerate(active)}
        valid_labels = np.array([remap_bgmm.get(l, -1) for l in valid_labels])

    elif algo == "otsu":
        from skimage.filters import threshold_otsu
        # threshold on PC1 (first column = dominant spectral axis)
        pc1 = pca_features[:, 0]
        thresh = threshold_otsu(pc1)
        valid_labels = (pc1 > thresh).astype(int)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose from: hdbscan, kmeans, gmm, agglomerative, optics")

    k = int(len([l for l in set(valid_labels) if l >= 0]))

    full_labels = np.full(len(masked_pixels), -2, dtype=int)
    full_labels[~specular_local] = valid_labels

    label_img = np.full((h, w), -2, dtype=int)
    label_img[mask_binary] = full_labels

    edge_band = int(MASK_EROSION_CLUSTERING + 5)
    mask_eroded_edge = ndimage.binary_erosion(mask_binary, iterations=edge_band)
    near_edge = mask_binary & ~mask_eroded_edge

    keep_ids = []
    cluster_ids = sorted([l for l in set(full_labels) if l >= 0])
    for cid in cluster_ids:
        cluster_pixels = label_img == cid
        total = int(np.sum(cluster_pixels))
        if total < MIN_CLUSTER_AREA_PX:
            continue
        edge_frac = float(np.sum(cluster_pixels & near_edge) / max(total, 1))
        if edge_frac > 0.50:
            label_img[cluster_pixels] = -1
            continue
        keep_ids.append(cid)

    unclustered = mask_binary & ~np.isin(label_img, keep_ids)
    unclust_count = int(np.sum(unclustered))
    if unclust_count >= MIN_NOISE_PROMOTE_PX:
        next_id = int((max(keep_ids) + 1) if keep_ids else 0)
        label_img[unclustered] = next_id
        keep_ids.append(next_id)

    keep_ids = list(dict.fromkeys(keep_ids))
    remap = {old: new for new, old in enumerate(keep_ids)}
    k = int(len(keep_ids))

    cmap = get_cmap(max(k, 1))
    label_colors = (cmap(np.arange(max(k, 1)))[:, :3] * 255).astype(np.uint8)

    out_img = np.zeros((h, w, 3), dtype=np.uint8)
    for old_id, new_id in remap.items():
        out_img[label_img == old_id] = label_colors[new_id]
    out_img[(label_img == -1) & mask_binary] = [40, 40, 40]

    vis = Image.fromarray(out_img)
    draw = ImageDraw.Draw(vis)
    targets = []
    for i in range(k):
        color_rgb = label_colors[i]
        cluster_mask = np.all(out_img == color_rgb, axis=2)
        cluster_inner = cluster_mask & mask_inner
        result = find_best_point(cluster_inner) or find_best_point(cluster_mask)
        if result:
            row, col, _area = result
            draw_crosshair(draw, col, row, size=10)
            draw.text((col + 5, row - 10), str(i), fill="white")
            targets.append({"cluster": int(i), "x": int(col), "y": int(row)})

    Image.fromarray(mask_u8).save(f"{output_prefix}_mask.png")
    Image.fromarray(out_img).save(f"{output_prefix}_clusters.png")
    with open(f"{output_prefix}_raman_targets.json", "w") as f:
        json.dump({"targets": targets}, f, indent=2)
    return k, len(targets)


def run_superpixel_clustering(cube, mask_u8, lab_image, output_prefix,
                               algorithm="hdbscan",
                               n_superpixels=200,
                               slic_compactness=5,
                               min_cluster_size=None, min_samples=None,
                               cluster_selection_epsilon=0.3,
                               cluster_selection_method="eom",
                               use_snv=True,
                               n_pca_components=None,
                               merge_min_fraction=0.0,
                               segment_method="slic",
                               segment_scale=100,
                               segment_sigma=0.8,
                               min_roi_pixels=1000):
    """
    SLIC superpixels on the reflectance LAB image, then cluster superpixels
    by their average HSI spectrum. Guarantees spatially compact regions.
    Saves {output_prefix}_clusters.png, _mask.png, _raman_targets.json.
    Returns (k, n_targets).
    """
    from skimage.segmentation import slic as slic_segment

    h, w, b = cube.shape
    mask_binary = mask_u8 > 127
    mask_inner = ndimage.binary_erosion(
        ndimage.binary_erosion(mask_binary, iterations=MASK_EROSION_CLUSTERING),
        iterations=max(0, MASK_EROSION_TARGETS - MASK_EROSION_CLUSTERING),
    )

    # Small ROI: skip clustering, treat the whole mask as one class
    roi_px = int(np.sum(mask_binary))
    if roi_px < min_roi_pixels:
        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        cmap = get_cmap(1)
        color = (np.array(cmap(0)[:3]) * 255).astype(np.uint8)
        out_img[mask_binary] = color
        vis = Image.fromarray(out_img)
        draw = ImageDraw.Draw(vis)
        targets = []
        result = find_best_point(mask_binary & mask_inner) or find_best_point(mask_binary)
        if result:
            row, col, _ = result
            draw_crosshair(draw, col, row, size=10)
            draw.text((col + 5, row - 10), "0", fill="white")
            targets.append({"cluster": 0, "x": int(col), "y": int(row)})
        Image.fromarray(mask_u8).save(f"{output_prefix}_mask.png")
        Image.fromarray(out_img).save(f"{output_prefix}_clusters.png")
        with open(f"{output_prefix}_raman_targets.json", "w") as f:
            json.dump({"targets": targets}, f, indent=2)
        return 1, len(targets)

    # Resize LAB image to cube spatial dims if needed
    if lab_image.shape[:2] != (h, w):
        lab_image = np.array(Image.fromarray(lab_image.astype(np.float32)).resize((w, h), Image.LANCZOS))

    # Oversegment the reflectance image into spatially coherent regions
    seg_method = segment_method.lower()
    if seg_method == "slic":
        segments = slic_segment(
            lab_image, n_segments=int(n_superpixels), compactness=float(slic_compactness),
            mask=mask_binary, start_label=1, channel_axis=2,
        )
    elif seg_method == "felzenszwalb":
        from skimage.segmentation import felzenszwalb
        segs = felzenszwalb(lab_image, scale=float(segment_scale), sigma=float(segment_sigma),
                            min_size=max(5, int(np.sum(mask_binary) // n_superpixels // 2)))
        segs[~mask_binary] = 0
        # relabel so background=0 and all segments within mask start from 1
        unique_ids = np.unique(segs[mask_binary])
        remap_seg = {old: new + 1 for new, old in enumerate(unique_ids)}
        segments = np.zeros_like(segs)
        for old, new in remap_seg.items():
            segments[segs == old] = new
    elif seg_method == "watershed":
        from skimage.segmentation import watershed
        from skimage.filters import sobel
        gray = lab_image[:, :, 0] / 100.0  # L channel normalised
        gradient = sobel(gray)
        from scipy.ndimage import distance_transform_edt, label as ndi_label
        distance = distance_transform_edt(mask_binary)
        # place one marker per expected region
        from skimage.feature import peak_local_max
        spacing = max(3, int(np.sqrt(np.sum(mask_binary) / n_superpixels)))
        coords = peak_local_max(distance, min_distance=spacing, labels=mask_binary)
        marker_mask = np.zeros(distance.shape, dtype=bool)
        marker_mask[tuple(coords.T)] = True
        markers, _ = ndi_label(marker_mask)
        segments = watershed(gradient, markers, mask=mask_binary)
        segments[~mask_binary] = 0
    else:
        raise ValueError(f"segment_method must be slic, felzenszwalb, or watershed. Got: {seg_method!r}")

    # Compute average HSI spectrum per superpixel
    sp_ids = [sid for sid in np.unique(segments) if sid > 0]
    sp_spectra, sp_ids_valid, sp_sizes = [], [], []
    for sp_id in sp_ids:
        sp_mask = (segments == sp_id) & mask_binary
        n_px = int(np.sum(sp_mask))
        if n_px < 3:
            continue
        sp_spectra.append(cube[sp_mask].mean(axis=0).astype(np.float32))
        sp_ids_valid.append(sp_id)
        sp_sizes.append(n_px)

    if len(sp_spectra) < 4:
        raise ValueError(f"Too few valid superpixels ({len(sp_spectra)})")

    sp_spectra = np.array(sp_spectra, dtype=np.float32)
    n_sp = len(sp_ids_valid)

    # SNV + PCA + StandardScaler on superpixel-averaged spectra
    spectra = apply_snv(sp_spectra) if use_snv else sp_spectra.copy()
    max_comp = int(min(25, spectra.shape[1], n_sp - 1))
    n_comp = int(min(n_pca_components, max_comp)) if n_pca_components else max_comp
    n_comp = max(n_comp, 1)
    pca = PCA(n_components=n_comp)
    features = pca.fit_transform(spectra)
    if n_pca_components is None:
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(max(2, int(np.searchsorted(cumvar, 0.95)) + 1))
        features = features[:, :min(n_keep, n_comp)]
    features = StandardScaler().fit_transform(features)

    # Cluster superpixels
    algo = algorithm.lower()
    if algo == "hdbscan":
        try:
            from hdbscan import HDBSCAN
        except Exception:
            from sklearn.cluster import HDBSCAN
        mcs = int(min_cluster_size) if min_cluster_size is not None else max(2, n_sp // 20)
        ms = int(min_samples) if min_samples is not None else max(1, n_sp // 50)
        try:
            sp_labels = HDBSCAN(
                min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=cluster_selection_epsilon,
                allow_single_cluster=True,
            ).fit_predict(features)
        except TypeError:
            sp_labels = HDBSCAN(
                min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=0.0,
                allow_single_cluster=True,
            ).fit_predict(features)
    elif algo == "bgmm":
        from sklearn.mixture import BayesianGaussianMixture
        bgmm = BayesianGaussianMixture(n_components=min(5, n_sp), random_state=42,
                                        n_init=3, weight_concentration_prior=1e-2)
        bgmm.fit(features)
        sp_labels = bgmm.predict(features)
        active = np.where(bgmm.weights_ > 0.02)[0]
        remap_b = {old: new for new, old in enumerate(active)}
        sp_labels = np.array([remap_b.get(l, -1) for l in sp_labels])
    elif algo == "meanshift":
        from sklearn.cluster import MeanShift, estimate_bandwidth
        bw = estimate_bandwidth(features, quantile=0.3, n_samples=min(500, n_sp))
        bw = bw if bw > 0 else 0.5
        sp_labels = MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(features)
    else:
        raise ValueError(f"Superpixel mode supports: hdbscan, bgmm, meanshift. Got: {algo!r}")

    # Merge minor clusters (below merge_min_fraction of total ROI pixels) into nearest large cluster
    if merge_min_fraction > 0:
        total_px = sum(sp_sizes)
        # Compute per-cluster pixel count and spectral centroid
        cluster_ids_raw = sorted([l for l in set(sp_labels) if l >= 0])
        cluster_px = {c: 0 for c in cluster_ids_raw}
        cluster_sum = {c: np.zeros(features.shape[1]) for c in cluster_ids_raw}
        for sp_id, label, feat in zip(sp_ids_valid, sp_labels, features):
            if label < 0:
                continue
            px = sp_sizes[sp_ids_valid.index(sp_id)]
            cluster_px[label] += px
            cluster_sum[label] += feat * px
        cluster_centroid = {c: cluster_sum[c] / max(cluster_px[c], 1) for c in cluster_ids_raw}
        major = [c for c in cluster_ids_raw if cluster_px[c] >= merge_min_fraction * total_px]
        minor = [c for c in cluster_ids_raw if c not in major]
        if minor and major:
            major_centroids = np.array([cluster_centroid[c] for c in major])
            remap_minor = {}
            for m in minor:
                dists = np.linalg.norm(major_centroids - cluster_centroid[m], axis=1)
                remap_minor[m] = major[int(np.argmin(dists))]
            sp_labels = np.array([remap_minor.get(l, l) for l in sp_labels])

    # Back-project superpixel labels to pixel image
    label_img = np.full((h, w), -1, dtype=int)
    for sp_id, label in zip(sp_ids_valid, sp_labels):
        label_img[segments == sp_id] = label

    # Build cluster list and remap to contiguous ids
    cluster_ids = sorted([l for l in set(sp_labels) if l >= 0])
    remap = {old: new for new, old in enumerate(cluster_ids)}
    k = len(cluster_ids)

    cmap = get_cmap(max(k, 1))
    label_colors = (cmap(np.arange(max(k, 1)))[:, :3] * 255).astype(np.uint8)

    out_img = np.zeros((h, w, 3), dtype=np.uint8)
    for old_id, new_id in remap.items():
        out_img[label_img == old_id] = label_colors[new_id]
    out_img[(label_img == -1) & mask_binary] = [40, 40, 40]

    vis = Image.fromarray(out_img)
    draw = ImageDraw.Draw(vis)
    targets = []
    for i in range(k):
        color_rgb = label_colors[i]
        cluster_mask = np.all(out_img == color_rgb, axis=2)
        cluster_inner = cluster_mask & mask_inner
        result = find_best_point(cluster_inner) or find_best_point(cluster_mask)
        if result:
            row, col, _area = result
            draw_crosshair(draw, col, row, size=10)
            draw.text((col + 5, row - 10), str(i), fill="white")
            targets.append({"cluster": int(i), "x": int(col), "y": int(row)})

    Image.fromarray(mask_u8).save(f"{output_prefix}_mask.png")
    Image.fromarray(out_img).save(f"{output_prefix}_clusters.png")
    with open(f"{output_prefix}_raman_targets.json", "w") as f:
        json.dump({"targets": targets}, f, indent=2)
    return k, len(targets)


def save_pca_preview(cube, output_path, n_components=3, sample_cap=150000, random_state=42):
    h, w, b = cube.shape
    vectors = cube.reshape(-1, b).astype(np.float32)
    keep = np.isfinite(vectors).all(axis=1)
    idx = np.where(keep)[0]
    if len(idx) == 0:
        Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(output_path)
        return
    if len(idx) > sample_cap:
        rng = np.random.default_rng(random_state)
        idx_s = rng.choice(idx, size=sample_cap, replace=False)
    else:
        idx_s = idx
    snv = apply_snv(vectors[idx_s])
    n_components = int(min(max(3, n_components), b, max(3, len(idx_s) // 25)))
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(snv)
    snv_all = apply_snv(vectors[idx])
    scores = pca.transform(snv_all)
    rgb = np.zeros((h * w, 3), dtype=np.uint8)
    for i in range(3):
        ch = np.zeros(h * w, dtype=np.float32)
        ch[idx] = scores[:, i].astype(np.float32)
        rgb[:, i] = normalize_minmax(ch.reshape(h, w)).reshape(-1)
    Image.fromarray(rgb.reshape(h, w, 3)).save(output_path)


def save_overlay(pca_rgb_path, mask_path, overlay_path):
    rgb = np.array(Image.open(pca_rgb_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    out = rgb.copy()
    out[mask] = (out[mask].astype(np.float32) * 0.5 + np.array([0.0, 255.0, 0.0]) * 0.5).astype(np.uint8)
    Image.fromarray(out).save(overlay_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--seg_k", type=int, default=5)
    parser.add_argument("--seg_pca", type=int, default=10)
    args = parser.parse_args()

    artifact_path = args.artifact
    artifact_name = os.path.basename(os.path.normpath(artifact_path))
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    output_prefix = os.path.join(out_dir, artifact_name)

    hsi_dir = os.path.join(artifact_path, "HSI")
    capture_path = find_capture_path(hsi_dir)
    if capture_path is None:
        raise FileNotFoundError(f"No HSI capture folder under {hsi_dir}")
    raw_path, hdr_path = find_hsi_files(capture_path)

    cube = read_hsi_cube(raw_path, hdr_path)
    save_pca_preview(cube, f"{output_prefix}_pca_map.jpg")

    mask = auto_mask_from_pca_kmeans(cube, n_components=args.seg_pca, k=args.seg_k)
    Image.fromarray(mask).save(f"{output_prefix}_mask.png")
    save_overlay(f"{output_prefix}_pca_map.jpg", f"{output_prefix}_mask.png", f"{output_prefix}_overlay.jpg")

    k, n_targets = run_hdbscan_clustering(cube, mask, output_prefix)
    print(f"{artifact_name}: mask + hdbscan saved ({k} clusters, {n_targets} targets)")


if __name__ == "__main__":
    main()

