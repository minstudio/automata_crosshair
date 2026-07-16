"""
analyze_hsi_clusters.py

For each HSI spectral cluster, map the 2D cluster region to the 3D mesh via
silhouette-based registration, then find the top 3 flattest non-overlapping
measurement positions (crosshairs) within that cluster's 3D region.

Clusters too small for at least --min-crosshairs placements are skipped.

Usage:
    python photogrammetry/analyze_hsi_clusters.py
    python photogrammetry/analyze_hsi_clusters.py --artifact lot1n1
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
import trimesh
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hsi"))
from paths import DATASET_ROOT
from hsi.hsi_3d_registration import (
    compute_flatness_scores,
    find_best_render_by_silhouette,
    load_viewfinder,
    look_at_pose,
    parse_datacube_angle,
    render_clay_views,
)
from hsi.test_pca_viewpoints import pca_viewpoints

# Camera intrinsics — must match test_2d_to_3d.py
YFOV  = np.pi / 3
IMG_W = IMG_H = 512
FY = FX = (IMG_H / 2) / np.tan(YFOV / 2)
CX, CY = IMG_W / 2, IMG_H / 2


# ── Geometry ──────────────────────────────────────────────────────────────────

def _cotlap(vertices, faces):
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0

    def cot(a, b):
        c = np.einsum("ij,ij->i", a, b)
        s = np.maximum(np.linalg.norm(np.cross(a, b), axis=1) ** 2, 1e-10)
        return np.clip(c / np.sqrt(s), -1e5, 1e5)

    c0, c1, c2 = cot(e1, e2), cot(e2, e0), cot(e0, e1)
    I = np.hstack([i1, i2, i2, i0, i0, i1])
    J = np.hstack([i2, i1, i0, i2, i1, i0])
    W = 0.5 * np.hstack([c0, c0, c1, c1, c2, c2])
    L = sp.coo_matrix((W, (I, J)), shape=(len(vertices), len(vertices)))
    diag = -np.asarray(L.sum(axis=1)).ravel()
    return 0.5 * (L + sp.diags(diag) + (L + sp.diags(diag)).T).tocsr()


def _smooth(vals, faces, n, iters):
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    I = np.hstack([i0, i1, i1, i2, i2, i0])
    J = np.hstack([i1, i0, i2, i1, i0, i2])
    A = sp.coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel() + 1.0
    r = vals.copy()
    for _ in range(iters):
        r = (r + A @ r) / deg
    return r


def compute_geometry(mesh):
    """Per-vertex normals, curvature (coarse + local), concavity depth."""
    V, F = mesh.vertices, mesh.faces
    n = len(V)

    fa = mesh.area_faces
    va = np.zeros(n)
    np.add.at(va, F[:, 0], fa / 3); np.add.at(va, F[:, 1], fa / 3); np.add.at(va, F[:, 2], fa / 3)

    tris = V[F]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    N = np.zeros_like(V)
    np.add.at(N, F[:, 0], fn); np.add.at(N, F[:, 1], fn); np.add.at(N, F[:, 2], fn)
    nrm = np.linalg.norm(N, axis=1)
    N[nrm > 0] /= nrm[nrm > 0, None]

    L = _cotlap(V, F)
    Lx = L @ V
    Lx /= np.maximum(va, 1e-20)[:, None]
    H_raw = 0.5 * np.linalg.norm(Lx, axis=1)
    H       = _smooth(H_raw, F, n, 10)
    H_local = _smooth(H_raw, F, n, 2)

    try:
        hull = ConvexHull(V)
        hn, ho = hull.equations[:, :3], hull.equations[:, 3]
        depths = np.zeros(n)
        for s in range(0, n, 10000):
            e = min(s + 10000, n)
            depths[s:e] = -(V[s:e] @ hn.T + ho).max(axis=1)
        depths = np.maximum(depths, 0.0)
        concave = _smooth(depths, F, n, 20)
        concave /= concave.max() + 1e-10
    except Exception as ex:
        print(f"  [!] ConvexHull failed: {ex}")
        concave = np.zeros(n)

    print(f"  Curvature range: {H_raw.min():.4f}–{H_raw.max():.4f} "
          f"(smoothed {H.min():.4f}–{H.max():.4f})")
    return N, H, H_local, concave, fa


# ── Cluster extraction ────────────────────────────────────────────────────────

def extract_clusters_from_png(clusters_img, min_blob_px=10):
    """
    Extract cluster masks directly from clusters.png without reanalysing colours.

    1. Quantise pixel colours to 4 levels per channel (floor to 64) so that
       anti-aliased fringe colours snap to the nearest true cluster colour.
    2. For each quantised colour do connected-component labelling separately.
    3. Drop components < min_blob_px  (removes the centroid + markers which
       are a handful of pixels).
    4. Merge CCs that share the same quantised colour.

    Returns list of (representative_color_rgb uint8[3], mask_bool H×W),
    sorted largest-first.
    """
    arr = np.array(clusters_img)
    is_fg = arr.sum(axis=2) > 50

    # Quantise: 0-63→0, 64-127→64, 128-191→128, 192-255→192
    q = (arr.astype(np.uint16) // 64 * 64).astype(np.uint8)
    q[~is_fg] = 0

    # Unique quantised colours (background = [0,0,0] already excluded via is_fg)
    packed = (q[:, :, 0].astype(np.uint32) << 16 |
              q[:, :, 1].astype(np.uint32) <<  8 |
              q[:, :, 2].astype(np.uint32))

    results = []
    for cval in np.unique(packed[is_fg]):
        if cval == 0:
            continue
        single = is_fg & (packed == cval)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            single.astype(np.uint8) * 255)
        combined = np.zeros(arr.shape[:2], dtype=bool)
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_px:
                combined |= (labels == lbl)
        if not combined.any():
            continue
        rep = np.median(arr[combined], axis=0).astype(np.uint8)
        results.append((rep, combined))

    results.sort(key=lambda x: -x[1].sum())
    return results


# ── 3-D registration helpers ──────────────────────────────────────────────────

def build_face_buffer(mesh_norm, cam_pose, grid=256):
    """Cast a grid of rays from the camera; return face-index map (grid×grid, -1=miss)."""
    u = np.linspace(0.5, IMG_W - 0.5, grid)
    v = np.linspace(0.5, IMG_H - 0.5, grid)
    UU, VV = np.meshgrid(u, v)
    uf, vf = UU.ravel(), VV.ravel()

    xc = (uf - CX) / FX
    yc = -(vf - CY) / FY
    zc = np.full_like(xc, -1.0)
    dirs = np.stack([xc, yc, zc], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_w = (cam_pose[:3, :3] @ dirs.T).T
    orig_w = np.tile(cam_pose[:3, 3], (len(uf), 1))

    print(f"  Casting {len(uf)} rays for face buffer...")
    face_hits, ray_hits = mesh_norm.ray.intersects_id(
        ray_origins=orig_w, ray_directions=dirs_w, multiple_hits=False)

    buf = np.full(grid * grid, -1, dtype=np.int32)
    if len(ray_hits):
        buf[ray_hits] = face_hits
    return buf.reshape(grid, grid)


def mask_to_face_indices(mask_cube, best_M, face_buf, grid=256):
    """
    Map a cube-space boolean mask → mesh face indices.

    Uses warpAffine to invert best_M on the whole mask at once, giving uniform
    render-space coverage without per-pixel gaps.  best_M maps render → cube
    space, so M_inv maps cube → render space.
    """
    vy, vx = np.where(mask_cube)
    if not len(vy):
        return np.array([], dtype=np.int32)

    if mask_cube.shape != (IMG_H, IMG_W):
        sy, sx = IMG_H / mask_cube.shape[0], IMG_W / mask_cube.shape[1]
        vy, vx = vy * sy, vx * sx

    M_inv = np.linalg.inv(best_M)
    pts  = np.stack([vx, vy, np.ones_like(vx)], axis=1).astype(float)
    rend = (M_inv @ pts.T).T
    rx   = rend[:, 0] / rend[:, 2]
    ry   = rend[:, 1] / rend[:, 2]

    gi = np.clip((ry / IMG_H * grid).astype(int), 0, grid - 1)
    gj = np.clip((rx / IMG_W * grid).astype(int), 0, grid - 1)
    fi  = face_buf[gi, gj]
    fi  = fi[fi >= 0]
    return np.unique(fi)


def project_norm_to_cube(pt_norm, cam_pose, best_M):
    """Forward-project a normalised-mesh 3D point to a cube-space (cx, cy) pixel
    (in IMG_W×IMG_H coords). Inverse of the cube→render→face path used elsewhere.
    Returns (cx, cy) or None if the point is behind the camera."""
    R, t = cam_pose[:3, :3], cam_pose[:3, 3]
    Pc = R.T @ (np.asarray(pt_norm, dtype=float) - t)   # world → camera
    if Pc[2] >= -1e-6:                                  # camera looks down -z
        return None
    u = CX + FX * (Pc[0] / -Pc[2])
    v = CY - FY * (Pc[1] / -Pc[2])
    cube = best_M @ np.array([u, v, 1.0])               # render → cube
    if abs(cube[2]) < 1e-9:
        return None
    return cube[0] / cube[2], cube[1] / cube[2]


def assign_xrf_to_clusters(xrf_results, cluster_list, cam_pose, best_M, mesh_norm,
                           close_ksize=15):
    """Map each XRF target to the cluster it lands in, so Raman can reuse it.

    An XRF target belongs to a cluster if its projected cube-space pixel falls
    inside that cluster's mask after morphological closing (bridges the dotted
    HSI pattern). Ties go to the cluster whose actual painted pixels are nearest.

    Returns {cluster_index: [xrf targets, in original XRF rank order]}.
    """
    assignment = {}
    if not xrf_results or not cluster_list:
        return assignment

    # Work in IMG_W×IMG_H space (best_M's cube space); resize masks to match.
    closed_dt = []
    for _color, m in cluster_list:
        mu = m.astype(np.uint8)
        if mu.shape != (IMG_H, IMG_W):
            mu = cv2.resize(mu, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        closed = cv2.morphologyEx(mu, cv2.MORPH_CLOSE,
                                  np.ones((close_ksize, close_ksize), np.uint8))
        dt = cv2.distanceTransform((mu == 0).astype(np.uint8), cv2.DIST_L2, 3)
        closed_dt.append((closed, dt))

    for rank, x in enumerate(xrf_results):
        proj = project_norm_to_cube(mesh_norm.vertices[x["idx"]], cam_pose, best_M)
        if proj is None:
            continue
        ix, iy = int(round(proj[0])), int(round(proj[1]))
        if not (0 <= ix < IMG_W and 0 <= iy < IMG_H):
            continue
        best_ci, best_d = None, np.inf
        for ci, (closed, dt) in enumerate(closed_dt):
            if closed[iy, ix] and dt[iy, ix] < best_d:
                best_ci, best_d = ci, dt[iy, ix]
        if best_ci is not None:
            x["xrf_rank"] = rank + 1
            assignment.setdefault(best_ci, []).append((rank, x))

    return {ci: [x for _r, x in sorted(lst, key=lambda rx: rx[0])]
            for ci, lst in assignment.items()}


# ── XRF square search ─────────────────────────────────────────────────────────

def find_xrf_squares(vertices, cand_idx, N, H_local, concave=None,
                     n_points=3, excluded=None, concave_penalty=50.0,
                     rim_tree=None, min_rim_dist=1.0):
    """
    Top n_points non-overlapping flat squares within cand_idx.
    Concavity penalty (convex-hull depth) strongly discourages concave surfaces.
    Candidates at least min_rim_dist (cm) from steep edges/rim (rim_tree) are
    preferred; the constraint is relaxed only if too few safe spots exist.
    Falls back through smaller scales if needed.
    Returns list of dicts: point, idx, h, rim_d, half_size, label.
    """
    search = cand_idx
    if len(search) > 1500:
        search = search[:: len(search) // 1500]

    rim_d_arr = None
    if rim_tree is not None and len(search):
        rim_d_arr, _ = rim_tree.query(vertices[search])

    scales = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025]
    best = []

    for scale in scales:
        half  = 0.5 * scale
        gap   = 3.0 * scale
        cands = []

        for si, vi in enumerate(search):
            pt = vertices[vi]
            nearby = np.where(np.sum((vertices - pt) ** 2, axis=1) < (half * 1.5) ** 2)[0]
            if not len(nearby):
                continue

            vn = N[vi]
            up = np.array([0., 1., 0.]) if abs(vn[1]) < 0.99 else np.array([1., 0., 0.])
            tx = np.cross(up, vn); tx /= np.linalg.norm(tx) + 1e-20
            ty = np.cross(vn, tx)

            pc = vertices[nearby] - pt
            in_sq = (np.abs(pc @ tx) <= half) & (np.abs(pc @ ty) <= half)
            if not in_sq.any():
                continue

            sq = nearby[in_sq]
            dots = N[sq] @ vn
            if dots.min() < 0.95:
                continue

            cv_penalty = concave[vi] * concave_penalty if concave is not None else 0.0
            score = H_local[sq].mean() + (1.0 - dots.mean()) * 100.0 + cv_penalty
            rim_d = float(rim_d_arr[si]) if rim_d_arr is not None else np.inf
            cands.append({"idx": vi, "point": pt.copy(), "h": score, "rim_d": rim_d})

        if not cands:
            continue

        # Graded score: normalised flatness + proportional rim-proximity penalty.
        # A candidate at min_rim_dist or further pays nothing; closer candidates
        # pay up to 2x the whole flatness range, so edge spots only win when
        # there is nothing meaningfully safer.
        hs = np.array([c["h"] for c in cands])
        h_lo, h_rng = hs.min(), max(float(np.ptp(hs)), 1e-9)
        for c in cands:
            rim_def = max(0.0, 1.0 - c["rim_d"] / min_rim_dist)
            c["score"] = (c["h"] - h_lo) / h_rng + 2.0 * rim_def

        cands.sort(key=lambda x: x["score"])
        labels = ["Rank 1 (Best)", "Rank 2", "Rank 3"]
        excl = excluded or []
        sel = []
        for c in cands:
            if len(sel) >= n_points:
                break
            if any(np.linalg.norm(c["point"] - s["point"]) < gap for s in sel + excl):
                continue
            c["label"] = labels[len(sel)] if len(sel) < len(labels) else f"Rank {len(sel)+1}"
            c["half_size"] = half
            sel.append(c)

        if len(sel) > len(best):
            best = sel
        if len(best) >= n_points:
            return best

    return best


# ── Crosshair drawing ─────────────────────────────────────────────────────────

# 5-row x 3-col pixel bitmaps for digits 1-3 (row 0=top, col 0=left)
_DIGIT_GLYPHS = {
    1: [(0,1),(1,0),(1,1),(2,1),(3,1),(4,1)],
    2: [(0,0),(0,1),(0,2),(1,2),(2,0),(2,1),(2,2),(3,0),(4,0),(4,1),(4,2)],
    3: [(0,0),(0,1),(0,2),(1,2),(2,1),(2,2),(3,2),(4,0),(4,1),(4,2)],
}


def _stamp_digit(vertices, N, pt, vi, half, colors, digit, color):
    """Stamp a pixel-art digit (1-3) centered at pt using the local tangent frame."""
    glyph = _DIGIT_GLYPHS.get(digit)
    if not glyph:
        return
    vn = N[vi]
    up = np.array([0., 1., 0.]) if abs(vn[1]) < 0.99 else np.array([1., 0., 0.])
    tx = np.cross(up, vn); tx /= np.linalg.norm(tx) + 1e-20
    ty = np.cross(vn, tx)
    cell   = max(half * 0.20, 0.012)
    nearby = np.where(np.linalg.norm(vertices - pt, axis=1) < half * 2.5)[0]
    if not len(nearby):
        return
    pc   = vertices[nearby] - pt
    lx   = pc @ tx
    ly   = pc @ ty
    lz   = np.abs(pc @ vn)
    z_ok = lz < half * 0.5
    n_ok = (N[nearby] @ vn) > 0.85
    half_cell = cell * 0.48
    for row, col in glyph:
        cx = (col - 1.0) * cell          # col 0->-cell, 1->0, 2->+cell
        cy = (2.0 - row) * cell          # row 0->+2*cell (top), row 4->-2*cell (bottom)
        in_cell = (np.abs(lx - cx) < half_cell) & (np.abs(ly - cy) < half_cell)
        colors[nearby[in_cell & z_ok & n_ok]] = color


def draw_crosshair(vertices, N, xrf, colors, color, label_num=None):
    """Square outline + crosshair for XRF targets. Only draws on flat-facing vertices."""
    pt        = xrf["point"]
    half      = xrf["half_size"]
    thickness = 0.04 * (half / 0.5)
    vi        = xrf["idx"]

    nearby = np.where(np.linalg.norm(vertices - pt, axis=1) < half * 2.5)[0]
    if not len(nearby):
        return

    vn = N[vi]
    up = np.array([0., 1., 0.]) if abs(vn[1]) < 0.99 else np.array([1., 0., 0.])
    tx = np.cross(up, vn); tx /= np.linalg.norm(tx) + 1e-20
    ty = np.cross(vn, tx)

    pc  = vertices[nearby] - pt
    ax, ay, az = np.abs(pc @ tx), np.abs(pc @ ty), np.abs(pc @ vn)

    n_ok    = (N[nearby] @ vn) > 0.85          # only flat-facing vertices
    z_ok    = az < half * 0.5
    flat    = n_ok & z_ok
    outline = ((np.abs(ax - half) < thickness) & (ay <= half + thickness)) | \
              ((np.abs(ay - half) < thickness) & (ax <= half + thickness))
    cross   = ((ay < thickness) & (ax < half * 2.0)) | \
              ((ax < thickness) & (ay < half * 2.0))
    colors[nearby[(outline | cross) & flat]] = color

    if label_num is not None:
        # Bright crosshair color -> black digit, dark -> white
        digit_color = np.array([20, 20, 20] if color.mean() > 127 else [255, 255, 255],
                               dtype=np.uint8)
        _stamp_digit(vertices, N, pt, vi, half, colors, label_num, digit_color)


# ── Raman component search ────────────────────────────────────────────────────

def _px_to_vertex(sx, sy, M_inv, face_buf, mesh_cm, grid, valid_fi=None):
    """Map one cube-space pixel → (vertex_idx, face_idx) via the face buffer.
    Only accepts face hits that are in valid_fi (the cluster's painted face set).
    Returns (vi, fi) or None."""
    pt_rend = M_inv @ np.array([sx, sy, 1.0], dtype=float)
    rx = pt_rend[0] / pt_rend[2]
    ry = pt_rend[1] / pt_rend[2]
    gi = int(np.clip(ry / IMG_H * grid, 0, grid - 1))
    gj = int(np.clip(rx / IMG_W * grid, 0, grid - 1))
    fi = face_buf[gi, gj]
    if fi < 0:
        return None
    if valid_fi is not None and fi not in valid_fi:
        return None
    fv = mesh_cm.vertices[mesh_cm.faces[fi]]
    fc = fv.mean(axis=0)
    vi = mesh_cm.faces[fi][np.argmin(np.linalg.norm(fv - fc, axis=1))]
    return int(vi), int(fi)


def _blob_centroid_to_3d(cx, cy, blob_yx, M_inv, face_buf, mesh_cm, vertices, grid,
                         valid_fi=None):
    """Map a 2D blob centroid (or nearest pixel) to a 3D vertex.
    Returns (face_vi, fi) or None."""
    search_pts = [[cy, cx]] + sorted(
        blob_yx.tolist(), key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
    for sy, sx in search_pts:
        r = _px_to_vertex(sx, sy, M_inv, face_buf, mesh_cm, grid, valid_fi)
        if r is not None:
            return r
    return None


def find_raman_targets(vertices, N, mask_cube, best_M, face_buf, mesh_cm,
                       n_points=3, min_points=2, grid=256, valid_fi=None,
                       H_local=None, rim_tree=None,
                       min_rim_dist=1.0, other_mask=None, min_other_dist_px=16,
                       max_cands_per_blob=80, excluded=None, target_gap=0.8):
    """
    Find Raman targets from the n_points largest 2D blobs in the cluster mask.
    Large single blobs are spatially subdivided so that n_points targets can be
    placed even in a fully contiguous region.

    Within each blob, interior pixels are mapped to the mesh and scored by
    normalised local flatness (H_local, with a mild pull toward the blob
    centroid) plus graded penalties for proximity to steep edges/rim (within
    min_rim_dist cm) and to other clusters (within min_other_dist_px cube px).
    The penalties are proportional, so even a blob that sits entirely on the
    rim places its point at the safest flat spot rather than ignoring the
    constraint. excluded is a list of already-placed 3D points (e.g. other
    clusters' targets) to keep a minimum spacing from.
    """
    mask_u8 = mask_cube.astype(np.uint8) * 255
    n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)
    comp_order = sorted(range(1, n_comp), key=lambda i: -stats[i, cv2.CC_STAT_AREA])

    rank_labels = ["Rank 1 (Biggest)", "Rank 2", "Rank 3"]
    half = 0.35
    M_inv = np.linalg.inv(best_M)

    # Distance (px) from any pixel to the other clusters' regions.
    # Close the masks first so dotted/sparse cluster paint counts as one region.
    d_other = None
    if other_mask is not None and other_mask.any():
        om = cv2.morphologyEx(other_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((7, 7), np.uint8))
        d_other = cv2.distanceTransform((om == 0).astype(np.uint8), cv2.DIST_L2, 3)

    # Collect candidate centroids from blobs; subdivide largest blob if needed
    candidates = []  # list of (cy, cx, n_px)
    for lbl in comp_order:
        candidates.append((centroids[lbl][1], centroids[lbl][0],
                           int(stats[lbl, cv2.CC_STAT_AREA]),
                           np.column_stack(np.where(labels == lbl))))

    # If we have fewer natural blobs than n_points, subdivide the largest blob
    if len(candidates) < n_points and candidates:
        big_yx = candidates[0][3]          # pixel positions of biggest blob
        need   = n_points - len(candidates) + 1  # how many sub-regions to cut it into
        # Sort along principal axis via PCA on pixel positions
        pts = big_yx.astype(float)
        pts -= pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        proj = big_yx.astype(float) @ vt[0]  # project onto principal axis
        edges = np.linspace(proj.min(), proj.max(), need + 1)
        sub_candidates = []
        for k in range(need):
            mask_seg = (proj >= edges[k]) & (proj < edges[k + 1])
            seg_yx = big_yx[mask_seg]
            if not len(seg_yx):
                continue
            cy_s, cx_s = seg_yx[:, 0].mean(), seg_yx[:, 1].mean()
            sub_candidates.append((cy_s, cx_s, int(len(seg_yx)), seg_yx))
        # Replace the first candidate with sub-regions
        candidates = sub_candidates + candidates[1:]

    targets = []
    # Spacing between targets (cm), center-to-center. The plus markers are
    # 2*half (~0.7 cm) wide, so target_gap only needs a small margin over that —
    # just enough to stop markers overlapping, not so much it shoves rank 2/3
    # out of their cluster centres.
    gap = max(target_gap, 2.1 * half)
    for cy, cx, n_px, blob_yx in candidates:
        if len(targets) >= n_points:
            break
        # Interior depth: keep candidates well inside the blob (close dotted masks)
        bm = np.zeros(mask_cube.shape, dtype=np.uint8)
        bm[blob_yx[:, 0].astype(int), blob_yx[:, 1].astype(int)] = 1
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        d_in  = cv2.distanceTransform(bm, cv2.DIST_L2, 3)
        depth = d_in[blob_yx[:, 0].astype(int), blob_yx[:, 1].astype(int)]
        core  = depth >= min(depth.max(), max(1.0, 0.4 * depth.max()))
        px    = blob_yx[core] if core.any() else blob_yx
        if len(px) > max_cands_per_blob:
            px = px[np.linspace(0, len(px) - 1, max_cands_per_blob).astype(int)]

        scored = []
        for sy, sx in px:
            r = _px_to_vertex(sx, sy, M_inv, face_buf, mesh_cm, grid, valid_fi)
            if r is None:
                continue
            vi, _ = r
            p3   = vertices[vi]
            flat = float(H_local[vi]) if H_local is not None else 0.0
            flat += 0.03 * float(np.hypot(sy - cy, sx - cx))  # stay near centroid
            rim_def = 0.0
            if rim_tree is not None:
                rim_def = max(0.0, 1.0 - float(rim_tree.query(p3)[0]) / min_rim_dist)
            oth_def = 0.0
            if d_other is not None:
                oth_def = max(0.0, 1.0 - float(d_other[int(sy), int(sx)]) / min_other_dist_px)
            scored.append({"vi": vi, "point": p3.copy(), "flat": flat,
                           "rim_def": rim_def, "oth_def": oth_def})

        # Graded score: normalised flatness + rim penalty + other-cluster penalty
        if scored:
            flats = np.array([c["flat"] for c in scored])
            f_lo, f_rng = flats.min(), max(float(np.ptp(flats)), 1e-9)
            for c in scored:
                c["score"] = ((c["flat"] - f_lo) / f_rng
                              + 2.0 * c["rim_def"] + 1.0 * c["oth_def"])
            scored.sort(key=lambda c: c["score"])

        avoid = [t["point"] for t in targets] + list(excluded or [])
        chosen = next((c for c in scored
                       if all(np.linalg.norm(c["point"] - p) >= gap
                              for p in avoid)), None)

        if chosen is not None:
            face_vi, pt = chosen["vi"], chosen["point"]
        else:
            # Fallback: original centroid mapping — but still respect spacing, so a
            # new target never lands on a reused XRF target or another target.
            result = _blob_centroid_to_3d(cx, cy, blob_yx, M_inv, face_buf,
                                          mesh_cm, vertices, grid, valid_fi=valid_fi)
            if result is None:
                continue
            face_vi, _ = result
            pt = vertices[face_vi].copy()
            if any(np.linalg.norm(pt - p) < gap for p in avoid):
                continue   # whole blob too close to an existing/reused target — skip

        targets.append({
            "idx": face_vi,
            "point": pt,
            "n_px": n_px,
            "label": (rank_labels[len(targets)] if len(targets) < len(rank_labels)
                      else f"Rank {len(targets)+1}"),
            "half_size": half,
        })

    if len(targets) < min_points:
        return []
    return targets


def draw_plus(vertices, N, target, colors, color, outline_color, label_num=None):
    """Draw a plus sign on top of the already-painted cluster region."""
    pt        = target["point"]
    half      = target["half_size"]
    thickness = 0.04 * (half / 0.35)
    vi        = target["idx"]

    nearby = np.where(np.linalg.norm(vertices - pt, axis=1) < half * 2.0)[0]
    if not len(nearby):
        return

    vn = N[vi]
    up = np.array([0., 1., 0.]) if abs(vn[1]) < 0.99 else np.array([1., 0., 0.])
    tx = np.cross(up, vn); tx /= np.linalg.norm(tx) + 1e-20
    ty = np.cross(vn, tx)

    pc  = vertices[nearby] - pt
    ax, ay, az = np.abs(pc @ tx), np.abs(pc @ ty), np.abs(pc @ vn)

    z_ok        = az < half * 0.5
    cross       = ((ay < thickness)       & (ax < half)) | ((ax < thickness)       & (ay < half))
    cross_thick = ((ay < thickness * 2.5) & (ax < half)) | ((ax < thickness * 2.5) & (ay < half))
    colors[nearby[cross_thick & z_ok]] = outline_color
    colors[nearby[cross       & z_ok]] = color

    if label_num is not None:
        # White outline -> black digit, dark outline -> white digit
        digit_color = np.array([20, 20, 20] if outline_color.mean() > 127 else [255, 255, 255],
                               dtype=np.uint8)
        _stamp_digit(vertices, N, pt, vi, half, colors, label_num, digit_color)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact",      default="lot1n1")
    ap.add_argument("--clusters-dir",  default=None,
                    help="Directory with *_clusters.png")
    ap.add_argument("--output",        default=None)
    ap.add_argument("--grid-size",     type=int, default=256,
                    help="Ray-grid resolution for face buffer")
    ap.add_argument("--min-crosshairs", type=int, default=2,
                    help="Skip cluster if fewer crosshairs fit")
    ap.add_argument("--xrf-only", action="store_true",
                    help="Skip Raman step (faster iteration on XRF only)")
    ap.add_argument("--rim-dist", type=float, default=1.0,
                    help="Preferred min distance (cm) from steep edges/rim (soft constraint)")
    ap.add_argument("--rim-angle", type=float, default=55.0,
                    help="Surface tilt (deg, vs viewing direction) above which "
                         "an area counts as steep edge/rim")
    ap.add_argument("--cluster-gap-px", type=int, default=16,
                    help="Preferred min distance (cube px) between a Raman target "
                         "and other clusters (graded soft constraint)")
    ap.add_argument("--raman-gap", type=float, default=0.8,
                    help="Min spacing (cm) between Raman targets and reused XRF "
                         "points; just enough to stop the markers overlapping")
    ap.add_argument("--all", action="store_true",
                    help="Process every artifact that has a cluster map and a 3D OBJ "
                         "(ignores --artifact)")
    args = ap.parse_args()

    if args.all:
        run_all(args)
    else:
        process_artifact(args.artifact, args)


def run_all(args):
    """Process all artifacts that have both a *_clusters.png and a 3D OBJ."""
    script_dir = Path(__file__).resolve().parent.parent
    clusters_dir = Path(args.clusters_dir) if args.clusters_dir \
        else script_dir / "hsi" / "output" / "roi_clustered"
    suffix = "_clusters.png"
    arts = sorted(p.name[:-len(suffix)] for p in clusters_dir.glob(f"*{suffix}"))
    if not arts:
        print(f"ERROR: no *{suffix} found in {clusters_dir}"); sys.exit(1)

    print(f"Found {len(arts)} artifacts with cluster maps in {clusters_dir}\n")
    ok, skipped, failed = [], [], []
    for i, art in enumerate(arts, 1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(arts)}] Processing {art}")
        print("=" * 70)
        try:
            done = process_artifact(art, args)
            (ok if done else skipped).append(art)
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append(art)
            print(f"  !! FAILED {art}: {e}")

    print("\n" + "=" * 70)
    print(f"All done: {len(ok)} ok, {len(skipped)} skipped, {len(failed)} failed.")
    if skipped:
        print(f"  skipped (missing data): {', '.join(skipped)}")
    if failed:
        print(f"  failed (error):         {', '.join(failed)}")


def process_artifact(art, args):
    """Run the full crosshair pipeline for one artifact.
    Returns True on success, False if required inputs are missing (skip)."""
    root = Path(DATASET_ROOT)
    art_dir = root / art
    script_dir = Path(__file__).resolve().parent.parent

    clusters_dir = Path(args.clusters_dir) if args.clusters_dir \
        else script_dir / "hsi" / "output" / "roi_clustered"
    roi_dir  = script_dir / "hsi" / "output" / "roi"
    out_dir  = Path(args.output) if args.output \
        else script_dir / "photogrammetry" / "output" / "hsi_clusters"
    out_dir.mkdir(parents=True, exist_ok=True)

    clusters_png = clusters_dir / f"{art}_clusters.png"
    roi_mask_png = roi_dir      / f"{art}_mask.png"

    if not clusters_png.exists():
        print(f"  SKIP: {clusters_png} not found"); return False

    # ── Mesh ──────────────────────────────────────────────────────────────────
    photo_raw = art_dir / "photogrammetry" / "raw_data"
    obj_files = sorted(photo_raw.glob("*.obj"))
    if not obj_files:
        print(f"  SKIP: no .obj under {photo_raw}"); return False
    mesh_path = obj_files[0]
    print(f"\n[mesh] {mesh_path.name}")

    # ── HSI metadata ──────────────────────────────────────────────────────────
    hsi_raw   = art_dir / "HSI" / "raw_data"
    if not hsi_raw.exists():
        hsi_raw = art_dir / "HSI"
    xml_files = sorted((hsi_raw / "metadata").glob("*.xml"))
    xml_path  = str(xml_files[0]) if xml_files else None
    cube_angle = parse_datacube_angle(xml_path)
    print(f"[hsi]  cube_angle={cube_angle}")

    vf_files = list((hsi_raw / "results").glob("RGBVIEWFINDER_*.png"))
    vf_path  = str(vf_files[0]) if vf_files else None
    viewfinder_bgr = load_viewfinder(vf_path, xml_path=xml_path)
    viewfinder_bgr = cv2.resize(viewfinder_bgr, (IMG_W, IMG_H))

    # ROI mask in viewfinder space
    if roi_mask_png.exists():
        vf_mask = cv2.imread(str(roi_mask_png), cv2.IMREAD_GRAYSCALE)
        if cube_angle != 0:
            vf_mask = np.rot90(vf_mask, k=((-cube_angle) // 90) % 4).copy()
        vf_mask = cv2.resize(vf_mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        _, vf_mask = cv2.threshold(vf_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        print("  [!] ROI mask not found — using clusters.png nonzero region")
        c = np.array(Image.open(clusters_png).convert("RGB"))
        vf_mask = ((c.sum(axis=2) > 30).astype(np.uint8) * 255)
        vf_mask = cv2.resize(vf_mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

    # ── Load + scale mesh ──────────────────────────────────────────────────────
    print("\n[1] Loading mesh...")
    raw = trimesh.load(str(mesh_path))
    if isinstance(raw, trimesh.Scene):
        geoms = list(raw.geometry.values())
        mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    else:
        mesh = raw
    print(f"    {len(mesh.vertices)} verts  {len(mesh.faces)} faces")

    # Normalised copy for registration / ray-casting
    mesh_norm = mesh.copy()
    bb_min, bb_max = mesh_norm.bounds
    center   = (bb_min + bb_max) / 2.0
    half_ext = np.max(bb_max - bb_min) / 2.0
    mesh_norm.vertices = (mesh_norm.vertices - center) / half_ext

    # Centimetre-scaled copy for crosshair search
    mesh_cm = mesh.copy()
    max_dim = np.max(mesh_cm.bounds[1] - mesh_cm.bounds[0])
    if max_dim < 0.5:
        mesh_cm.vertices *= 100.0; print("    Units: m -> x100 cm")
    elif max_dim > 100:
        mesh_cm.vertices *= 0.1;   print("    Units: mm -> x0.1 cm")
    else:
        print("    Units: cm")

    # ── Geometry ───────────────────────────────────────────────────────────────
    print("\n[2] Computing geometry...")
    N_arr, H_arr, H_local, concave, face_areas = compute_geometry(mesh_cm)
    flatness = compute_flatness_scores(mesh_norm)

    # ── Registration ───────────────────────────────────────────────────────────
    print("\n[3] PCA viewpoints + silhouette matching...")
    vps     = pca_viewpoints(mesh_norm)
    renders = render_clay_views(mesh_norm, flatness, vps, size=IMG_W)
    c_arr = np.array(Image.open(clusters_png).convert("RGB"))
    cluster_sil = ((c_arr.sum(axis=2) > 30).astype(np.uint8) * 255)
    cluster_sil = cv2.resize(cluster_sil, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    best_idx, _, best_iou, best_M, _ = find_best_render_by_silhouette(cluster_sil, renders)
    best_eye = np.array(renders[best_idx]["eye"])
    cam_pose = look_at_pose(best_eye)
    print(f"    IoU={best_iou:.3f}  eye={best_eye.round(3)}")

    # Steep-edge / rim proximity: a vertex is "steep" when its smoothed normal
    # tilts more than rim_angle away from the viewing direction — that catches
    # the rim band, ridge flanks and fracture sides, but not surface texture.
    view_dir = best_eye / (np.linalg.norm(best_eye) + 1e-20)
    F = mesh_cm.faces
    n_v = len(mesh_cm.vertices)
    I = np.hstack([F[:, 0], F[:, 1], F[:, 1], F[:, 2], F[:, 2], F[:, 0]])
    J = np.hstack([F[:, 1], F[:, 0], F[:, 2], F[:, 1], F[:, 0], F[:, 2]])
    A = sp.coo_matrix((np.ones(len(I)), (I, J)), shape=(n_v, n_v)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel() + 1.0
    Ns = N_arr.copy()
    for _ in range(5):
        Ns = (Ns + A @ Ns) / deg[:, None]
    Ns /= np.linalg.norm(Ns, axis=1, keepdims=True) + 1e-20
    dots = Ns @ view_dir
    # Side-facing band only: exclude the back surface — on a thin sherd the
    # back is within ~thickness of every front point and would cap rim
    # distances at the sherd thickness, masking real edge proximity.
    steep_mask = (dots < np.cos(np.radians(args.rim_angle))) & (dots > -0.5)
    rim_tree = cKDTree(mesh_cm.vertices[steep_mask]) if steep_mask.any() else None
    print(f"    Steep-edge vertices: {int(steep_mask.sum())} "
          f"(tilt > {args.rim_angle:.0f} deg from view)")

    # ── Face buffer ────────────────────────────────────────────────────────────
    print(f"\n[4] Face index buffer ({args.grid_size}²)...")
    face_buf = build_face_buffer(mesh_norm, cam_pose, grid=args.grid_size)
    print(f"    {(face_buf >= 0).sum()} cells hit the mesh")

    # ── Cluster masks ──────────────────────────────────────────────────────────
    print("\n[5] Extracting cluster masks...")
    clusters_img = Image.open(clusters_png).convert("RGB")
    cluster_list = extract_clusters_from_png(clusters_img)
    methods = {"clusters": cluster_list}
    print(f"    {len(cluster_list)} clusters  "
          f"({', '.join(str(m.sum()) + 'px' for _, m in cluster_list)})")

    V_cm = mesh_cm.vertices

    # ── XRF: flattest squares among camera-visible faces (projection system) ────
    # Recycle the Raman face-buffer projection: XRF seeds are the vertices of the
    # faces the camera actually sees, so a crosshair can never land on the hidden
    # back surface (the old BFS flat-region search would happily pick the smoother
    # back of a thin sherd). find_xrf_squares then ranks by flatness + rim.
    print("\n[6/xrf] Visible-surface XRF search...")
    vis_faces = np.unique(face_buf[face_buf >= 0])
    if len(vis_faces):
        xrf_cand = np.unique(mesh_cm.faces[vis_faces].ravel())
    else:
        xrf_cand = np.arange(len(V_cm))
    print(f"  Searching {len(xrf_cand)} camera-visible vertices for XRF squares...")
    xrf_results = find_xrf_squares(V_cm, xrf_cand, N_arr, H_local,
                                   concave=concave, n_points=3, concave_penalty=50.0,
                                   rim_tree=rim_tree, min_rim_dist=args.rim_dist)
    for x in xrf_results:
        print(f"    {x['label']}  pt={x['point'].round(3)}"
              f"  half={x['half_size']:.3f}  h={x['h']:.4f}  rim_d={x['rim_d']:.2f}")

    # ── Raman: per-cluster connected-component targets ─────────────────────────
    if args.xrf_only:
        print("\n[6/raman] Skipped (--xrf-only)")
        raman_data = []
    else:
        print("\n[6/raman] Per-cluster Raman targets...")
        raman_data = []
        placed_pts = []   # 3D points already chosen, across clusters

        # Reuse XRF targets as Raman targets where an XRF point lands in a cluster.
        xrf_by_cluster = assign_xrf_to_clusters(xrf_results, cluster_list,
                                                cam_pose, best_M, mesh_norm)
        for ci, lst in xrf_by_cluster.items():
            ranks = ", ".join(str(x.get("xrf_rank", "?")) for x in lst)
            print(f"  Cluster {ci} reuses XRF rank(s): {ranks}")

        for ci, (color, mask_cube) in enumerate(cluster_list):
            print(f"\n  Cluster {ci}  color={color}  pixels={mask_cube.sum()}")
            fi = mask_to_face_indices(mask_cube, best_M, face_buf, grid=args.grid_size)
            if not len(fi):
                print("    SKIP — no mesh faces found")
                raman_data.append({"ci": ci, "color": color, "mask": mask_cube,
                                   "skipped": True, "reason": "no 3d faces", "targets": []})
                continue

            vi = np.unique(mesh_cm.faces[fi].ravel())
            fi_set = set(fi.tolist())   # cluster's painted face set

            # XRF targets inside this cluster become its first Raman targets
            # (renumbered sequentially); the rest are filled by the blob
            # heuristic, kept clear of the reused points.
            reused = [{"idx": x["idx"], "point": x["point"].copy(), "n_px": None,
                       "half_size": 0.35, "source": "xrf", "xrf_rank": x.get("xrf_rank")}
                      for x in xrf_by_cluster.get(ci, [])]
            n_remaining = 3 - len(reused)

            heuristic = []
            if n_remaining > 0:
                other_mask = np.zeros_like(mask_cube)
                for cj, (_c2, m2) in enumerate(cluster_list):
                    if cj != ci:
                        other_mask |= m2
                heuristic = find_raman_targets(
                    V_cm, N_arr, mask_cube, best_M, face_buf, mesh_cm,
                    n_points=n_remaining, min_points=0,
                    grid=args.grid_size, valid_fi=fi_set,
                    H_local=H_local, rim_tree=rim_tree, min_rim_dist=args.rim_dist,
                    other_mask=other_mask, min_other_dist_px=args.cluster_gap_px,
                    excluded=placed_pts + [r["point"] for r in reused],
                    target_gap=args.raman_gap)
                for t in heuristic:
                    t["source"] = "auto"

            targets = reused + heuristic
            if len(targets) < args.min_crosshairs:
                print(f"    SKIP — fewer than {args.min_crosshairs} viable targets")
                raman_data.append({"ci": ci, "color": color, "mask": mask_cube,
                                   "skipped": True, "reason": "insufficient components",
                                   "n_faces": int(len(fi)), "targets": []})
                continue

            # Renumber labels 1..k sequentially (reused XRF points first)
            for i, t in enumerate(targets):
                src = "XRF" if t.get("source") == "xrf" else "auto"
                t["label"] = f"Rank {i + 1} ({src})"

            placed_pts.extend(t["point"] for t in targets)
            print(f"    Raman targets: {len(targets)}  "
                  f"({len(reused)} from XRF, {len(heuristic)} auto)")
            for t in targets:
                print(f"      {t['label']}  pt={t['point'].round(3)}  px={t['n_px']}")

            raman_data.append({"ci": ci, "color": color, "mask": mask_cube,
                               "skipped": False, "n_faces": int(len(fi)), "n_verts": int(len(vi)),
                               "targets": targets})

    # ── PLY output ────────────────────────────────────────────────────────────
    print(f"\n[7] Saving PLY files to {out_dir}/")

    # Base: flatness heatmap — brighter = flatter
    f_norm = flatness
    base_gray = (f_norm * 170 + 50).astype(np.uint8)
    base_colors = np.stack([base_gray, base_gray, base_gray], axis=1)

    def _rank_colors(base_rgb, n):
        """(fill, outline) tuples: rank 1 = brightest/white, rank n = darkest/dark-grey."""
        b = np.array(base_rgb, dtype=float)
        scales   = np.linspace(1.0, 0.44, max(n, 1))
        grey_lvl = [255, 170, 85]
        fills    = [np.clip(b * s, 0, 255).astype(np.uint8) for s in scales]
        outlines = [np.full(3, grey_lvl[min(i, 2)], dtype=np.uint8) for i in range(n)]
        return list(zip(fills, outlines))

    # XRF PLY: flatness heatmap + global XRF squares (Red/Orange/Yellow per rank)
    XRF_COLORS = [
        np.array([255,   0,   0], dtype=np.uint8),  # rank 1 — red
        np.array([255, 165,   0], dtype=np.uint8),  # rank 2 — orange
        np.array([255, 255,   0], dtype=np.uint8),  # rank 3 — yellow
    ]
    xrf_colors = base_colors.copy()
    for rank, xrf in enumerate(xrf_results):
        draw_crosshair(V_cm, N_arr, xrf, xrf_colors, XRF_COLORS[rank], label_num=rank + 1)
    xrf_ply = out_dir / f"{art}_xrf.ply"
    trimesh.Trimesh(vertices=V_cm, faces=mesh_cm.faces,
                    vertex_colors=xrf_colors).export(str(xrf_ply))
    print(f"    {xrf_ply.name}  ({len(xrf_results)} XRF targets)")

    if not args.xrf_only:
        # Raman PLY: cluster colors + plus-sign targets
        raman_colors = base_colors.copy()
        for entry in raman_data:
            if not entry["skipped"]:
                fi = mask_to_face_indices(entry["mask"], best_M, face_buf, grid=args.grid_size)
                if len(fi):
                    vi = np.unique(mesh_cm.faces[fi].ravel())
                    raman_colors[vi] = entry["color"]
        for entry in raman_data:
            if not entry["skipped"]:
                rc = _rank_colors(entry["color"], len(entry["targets"]))
                for rank, (fill_c, outline_c) in enumerate(rc):
                    draw_plus(V_cm, N_arr, entry["targets"][rank], raman_colors, fill_c, outline_c, label_num=rank + 1)
        raman_ply = out_dir / f"{art}_raman.ply"
        trimesh.Trimesh(vertices=V_cm, faces=mesh_cm.faces,
                        vertex_colors=raman_colors).export(str(raman_ply))
        valid_r = sum(1 for e in raman_data if not e["skipped"])
        print(f"    {raman_ply.name}  ({valid_r} valid clusters)")

        # Regions-only PLY for validation
        reg_colors = base_colors.copy()
        for entry in raman_data:
            fi = mask_to_face_indices(entry["mask"], best_M, face_buf, grid=args.grid_size)
            if not len(fi):
                continue
            vi = np.unique(mesh_cm.faces[fi].ravel())
            reg_colors[vi] = entry["color"]
        reg_path = out_dir / f"{art}_hsi_clusters_regions.ply"
        trimesh.Trimesh(vertices=V_cm, faces=mesh_cm.faces,
                        vertex_colors=reg_colors).export(str(reg_path))
        print(f"    {reg_path.name}  (regions only)")

        # Debug cluster-mask PNG
        dbg = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        for color, mask in cluster_list:
            m = mask.astype(np.uint8)
            if m.shape != (IMG_H, IMG_W):
                m = cv2.resize(m, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
            dbg[m > 0] = color
        cv2.imwrite(str(out_dir / f"{art}_clusters_debug.png"),
                    cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))

        def _ser_raman(entry):
            return {
                "cluster": entry["ci"],
                "color": [int(c) for c in entry["color"]],
                "skipped": entry["skipped"],
                "reason": entry.get("reason"),
                "n_faces": entry.get("n_faces"),
                "targets": [{"label": t.get("label", "?"),
                             "point": [round(float(v), 4) for v in t["point"]],
                             "n_px": t.get("n_px"),
                             "source": t.get("source", "auto"),
                             "xrf_rank": t.get("xrf_rank")}
                            for t in entry.get("targets", [])],
            }

        raman_report = {
            "artifact": art,
            "mesh": str(mesh_path),
            "silhouette_iou": round(float(best_iou), 4),
            "results": [_ser_raman(e) for e in raman_data],
        }
        raman_rpt = out_dir / f"{art}_raman_targets.json"
        with open(raman_rpt, "w") as f:
            json.dump(raman_report, f, indent=2)
        print(f"    {raman_rpt.name}")

    # ── XRF JSON ───────────────────────────────────────────────────────────────
    xrf_report = {
        "artifact": art,
        "mesh": str(mesh_path),
        "silhouette_iou": round(float(best_iou), 4),
        "targets": [{"label": x.get("label", "?"),
                     "point": [round(float(v), 4) for v in x["point"]],
                     "half_size": round(float(x["half_size"]), 4),
                     "h": round(float(x["h"]), 6),
                     "rim_dist": (round(float(x["rim_d"]), 3)
                                  if np.isfinite(x.get("rim_d", np.inf)) else None)}
                    for x in xrf_results],
    }
    xrf_rpt = out_dir / f"{art}_xrf_report.json"
    with open(xrf_rpt, "w") as f:
        json.dump(xrf_report, f, indent=2)
    print(f"    {xrf_rpt.name}")

    print(f"\nDone.\n  XRF:   {xrf_ply}"
          + (f"\n  Raman: {raman_ply}" if not args.xrf_only else "  (Raman unchanged)"))
    return True


if __name__ == "__main__":
    main()
