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
from scipy.spatial import ConvexHull

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


# ── BFS flat-region detection (ported from analyze_3d_zecchin_old.py) ─────────

def _vertex_adjacency(faces, n):
    """CSR adjacency list: adjacency[i] = array of neighbour vertex indices."""
    I = np.hstack([faces[:, 0], faces[:, 1], faces[:, 1],
                   faces[:, 2], faces[:, 2], faces[:, 0]])
    J = np.hstack([faces[:, 1], faces[:, 0], faces[:, 2],
                   faces[:, 1], faces[:, 0], faces[:, 2]])
    A = sp.coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)).tocsr()
    return [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(n)]


def _vertex_to_faces(faces, n):
    """For each vertex, list of face indices that touch it."""
    fi = np.arange(len(faces))
    I  = np.hstack([faces[:, 0], faces[:, 1], faces[:, 2]])
    J  = np.hstack([fi, fi, fi])
    M  = sp.coo_matrix((np.ones(len(I)), (I, J)), shape=(n, len(faces))).tocsr()
    return [M.indices[M.indptr[i]:M.indptr[i + 1]] for i in range(n)]


def find_flat_regions(vertices, faces, face_areas, N, H, concave,
                      curvature_percent=50.0,
                      normal_similarity=0.50,
                      min_region_vertex_count=30,
                      target_sensor_area_sq_cm=2.0,
                      curvature_tolerance_factor=10.0,
                      concave_penalty_factor=50.0):
    """
    BFS region grower that finds macroscopic flat convex patches.
    Seeds from the lowest-curvature vertices; expands while neighbours
    agree in normal direction and curvature. Scores each region by
    curvature + area bonus + concavity penalty (same as old script).
    Returns vertex indices of the best (most convex, flattest, largest) region.
    """
    from collections import deque

    n = len(vertices)
    k = max(1, int(n * curvature_percent / 100))
    low_curv_idx = np.argpartition(H, k)[:k]

    adjacency      = _vertex_adjacency(faces, n)
    vtf            = _vertex_to_faces(faces, n)

    visited  = np.zeros(n, dtype=bool)
    regions  = []

    for seed in low_curv_idx:
        if visited[seed]:
            continue

        queue = deque([seed])
        visited[seed] = True

        max_h   = H[seed] * (1.0 + curvature_tolerance_factor)
        verts   = {seed}
        f_set   = set(vtf[seed].tolist())
        n_sum   = N[seed].copy()

        while queue:
            v = queue.popleft()
            for nb in adjacency[v]:
                if visited[nb]:
                    continue
                if H[nb] > max_h:
                    continue
                n_avg = n_sum / (np.linalg.norm(n_sum) + 1e-20)
                if np.dot(N[nb], n_avg) < normal_similarity:
                    continue
                visited[nb] = True
                verts.add(nb)
                queue.append(nb)
                n_sum += N[nb]
                f_set.update(vtf[nb].tolist())

        if len(verts) < min_region_vertex_count:
            continue

        va = np.array(list(verts))
        area = sum(face_areas[fi] for fi in f_set
                   if all(v in verts for v in faces[fi]))
        if area <= 0:
            continue

        # Aspect-ratio filter (reject thin strips)
        pts  = vertices[va]
        ptsc = pts - pts.mean(axis=0)
        nrm  = n_sum / (np.linalg.norm(n_sum) + 1e-20)
        up   = np.array([0., 1., 0.]) if abs(nrm[1]) < 0.99 else np.array([1., 0., 0.])
        tx   = np.cross(up, nrm); tx /= np.linalg.norm(tx) + 1e-20
        ty   = np.cross(nrm, tx)
        c2d  = np.column_stack([ptsc @ tx, ptsc @ ty])
        ev   = np.maximum(np.linalg.eigvalsh(np.cov(c2d, rowvar=False)), 1e-20)
        if np.sqrt(ev.min()) / (np.sqrt(ev.max()) + 1e-20) < 0.05:
            continue

        regions.append({
            "vertices":     va,
            "area":         area,
            "normal":       nrm,
            "curvature_H":  H[va].mean(),
            "concave_ratio": concave[va].mean(),
        })

    if not regions:
        return np.array([], dtype=np.int32)

    # Score: normalised curvature – area_bonus×20 + concavity_penalty×factor
    curvs  = np.array([r["curvature_H"]  for r in regions])
    areas  = np.array([r["area"]          for r in regions])
    c_min, c_max = curvs.min(), curvs.max()
    a_min, a_max = areas.min(), areas.max()

    # Effective area threshold (lower if nothing big enough)
    thr = target_sensor_area_sq_cm
    if areas.max() < thr:
        thr = areas.max() * 0.5
        print(f"  [!] No region >= {target_sensor_area_sq_cm:.2f} sq cm; "
              f"lowering threshold to {thr:.4f} sq cm")

    valid = [r for r in regions if r["area"] >= thr]
    if not valid:
        valid = regions  # fallback: use all

    curvs  = np.array([r["curvature_H"]  for r in valid])
    areas  = np.array([r["area"]          for r in valid])
    c_min, c_max = curvs.min(), curvs.max()
    a_min, a_max = areas.min(), areas.max()

    for i, r in enumerate(valid):
        c_norm = (curvs[i] - c_min) / (c_max - c_min + 1e-20)
        a_norm = (areas[i] - a_min) / (a_max - a_min + 1e-20)
        r["score"] = c_norm - 20.0 * a_norm + concave_penalty_factor * r["concave_ratio"]

    valid.sort(key=lambda r: r["score"])
    best = valid[0]
    print(f"  BFS regions found: {len(regions)}  valid (>= area thr): {len(valid)}")
    print(f"  Best region: {len(best['vertices'])} verts  "
          f"area={best['area']:.2f} sq cm  "
          f"curv={best['curvature_H']:.4f}  "
          f"concave={best['concave_ratio']:.4f}  "
          f"score={best['score']:.4f}")
    return best["vertices"]


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


# ── XRF square search ─────────────────────────────────────────────────────────

def find_xrf_squares(vertices, cand_idx, N, H_local, concave=None,
                     n_points=3, excluded=None, concave_penalty=50.0, 
                     min_edge_distance_factor=2.5):
    """
    Top n_points non-overlapping flat squares within cand_idx.
    Concavity penalty (convex-hull depth) strongly discourages concave surfaces.
    Falls back through smaller scales if needed.
    
    min_edge_distance_factor: Minimum distance from region edge as multiple 
                              of crosshair half_size (e.g., 2.5 = 2.5 crosshair widths)
    
    Returns list of dicts: point, idx, h, half_size, label.
    """
    search = cand_idx
    if len(search) > 1500:
        search = search[:: len(search) // 1500]
    
    # Compute boundary vertices (vertices on edge of candidate region)
    # Find vertices that have neighbors outside the candidate region
    from scipy.spatial.distance import cdist
    all_vertices = np.arange(len(vertices))
    boundary_vertices = []
    
    # For efficiency, use spatial distance to find boundary
    if len(cand_idx) > 0:
        cand_set = set(cand_idx)
        for vi in cand_idx:
            # Find nearby vertices within a small radius
            distances = np.sqrt(np.sum((vertices - vertices[vi])**2, axis=1))
            nearby_idx = np.where(distances < 0.5)[0]  # Within 0.5cm
            
            # If any nearby vertices are outside candidate region, this is boundary
            if any(idx not in cand_set for idx in nearby_idx):
                boundary_vertices.append(vi)
        
        boundary_vertices = np.array(boundary_vertices)

    scales = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025]
    best = []

    for scale in scales:
        half  = 0.5 * scale
        gap   = 3.0 * scale
        cands = []

        for vi in search:
            pt = vertices[vi]
            
            # Check distance from region boundary
            min_distance_required = half * min_edge_distance_factor
            if len(boundary_vertices) > 0:
                boundary_distances = np.sqrt(np.sum((vertices[boundary_vertices] - pt)**2, axis=1))
                min_boundary_distance = boundary_distances.min()
                if min_boundary_distance < min_distance_required:
                    continue  # Too close to edge
            
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
            cands.append({"idx": vi, "point": pt.copy(), "h": score})

        cands.sort(key=lambda x: x["h"])
        labels = ["Rank 1 (Best)", "Rank 2", "Rank 3"]
        sel = []
        for c in cands:
            if len(sel) >= n_points:
                break
            excl = excluded or []
            if not any(np.linalg.norm(c["point"] - s["point"]) < gap for s in sel + excl):
                c["label"] = labels[len(sel)]
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

def _blob_centroid_to_3d(cx, cy, blob_yx, M_inv, face_buf, mesh_cm, vertices, grid,
                         valid_fi=None):
    """Map a 2D blob centroid (or nearest pixel) to a 3D vertex.
    Only accepts face hits that are in valid_fi (the cluster's painted face set).
    Returns (face_vi, fi) or None."""
    search_pts = [[cy, cx]] + sorted(
        blob_yx.tolist(), key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
    for sy, sx in search_pts:
        pt_rend = M_inv @ np.array([sx, sy, 1.0], dtype=float)
        rx = pt_rend[0] / pt_rend[2]
        ry = pt_rend[1] / pt_rend[2]
        gi = int(np.clip(ry / IMG_H * grid, 0, grid - 1))
        gj = int(np.clip(rx / IMG_W * grid, 0, grid - 1))
        fi = face_buf[gi, gj]
        if fi < 0:
            continue
        if valid_fi is not None and fi not in valid_fi:
            continue
        fv = mesh_cm.vertices[mesh_cm.faces[fi]]
        fc = fv.mean(axis=0)
        vi = mesh_cm.faces[fi][np.argmin(np.linalg.norm(fv - fc, axis=1))]
        return int(vi), int(fi)
    return None


def find_raman_targets(vertices, N, mask_cube, best_M, face_buf, mesh_cm,
                       n_points=3, min_points=2, grid=256, valid_fi=None):
    """
    Find Raman targets from the n_points largest 2D blobs in the cluster mask.
    Large single blobs are spatially subdivided so that n_points targets can be
    placed even in a fully contiguous region.
    Maps each blob centroid from cube space → render space → mesh face → 3D point.
    """
    mask_u8 = mask_cube.astype(np.uint8) * 255
    n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)
    comp_order = sorted(range(1, n_comp), key=lambda i: -stats[i, cv2.CC_STAT_AREA])

    rank_labels = ["Rank 1 (Biggest)", "Rank 2", "Rank 3"]
    half = 0.35
    M_inv = np.linalg.inv(best_M)

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
    for i, (cy, cx, n_px, blob_yx) in enumerate(candidates[:n_points]):
        result = _blob_centroid_to_3d(cx, cy, blob_yx, M_inv, face_buf, mesh_cm, vertices, grid,
                                      valid_fi=valid_fi)
        if result is None:
            continue
        face_vi, _ = result
        targets.append({
            "idx": face_vi,
            "point": vertices[face_vi].copy(),
            "n_px": n_px,
            "label": rank_labels[i] if i < len(rank_labels) else f"Rank {i+1}",
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
    ap.add_argument("--grid-size",     type=int, default=128,
                    help="Ray-grid resolution for face buffer (lower=faster)")
    ap.add_argument("--min-crosshairs", type=int, default=2,
                    help="Skip cluster if fewer crosshairs fit")
    ap.add_argument("--xrf-only", action="store_true",
                    help="Skip Raman step (faster iteration on XRF only)")
    ap.add_argument("--low-memory", action="store_true",
                    help="Use low-memory settings (faster, lower quality)")
    ap.add_argument("--edge-distance", type=float, default=2.5,
                    help="Minimum distance from region edge as multiple of crosshair size (default: 2.5)")
    args = ap.parse_args()

    art = args.artifact
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
        print(f"ERROR: {clusters_png} not found"); sys.exit(1)

    # ── Mesh ──────────────────────────────────────────────────────────────────
    photo_raw = art_dir / "photogrammetry" / "raw_data"
    obj_files = sorted(photo_raw.glob("*.obj"))
    if not obj_files:
        print(f"ERROR: no .obj under {photo_raw}"); sys.exit(1)
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
    
    # Simplify mesh if too large (reduce memory usage)
    max_faces = 50000 if args.low_memory else 200000  # Lower limit for low-memory mode
    if len(mesh.faces) > max_faces:
        print(f"    Mesh too large ({len(mesh.faces)} faces), simplifying to {max_faces} faces...")
        original_faces = len(mesh.faces)
        # Use progressive mesh simplification that preserves topology
        try:
            # Try to install and use pymeshlab for better decimation
            import pymeshlab as ml
            ms = ml.MeshSet()
            ms.add_mesh(ml.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
            
            # Calculate target number of faces
            target_faces = max_faces
            ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=target_faces)
            
            # Get simplified mesh
            simplified = ms.current_mesh()
            mesh = trimesh.Trimesh(vertices=simplified.vertex_matrix(), faces=simplified.face_matrix())
            print(f"    Used pymeshlab quadric edge collapse")
            
        except ImportError:
            # Fallback: use trimesh built-in simplification
            try:
                # Try basic trimesh simplification
                mesh = mesh.smoothed()  # Smooth first to help with decimation
                
                # Remove duplicate vertices and degenerate faces
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
                mesh.remove_unreferenced_vertices()
                
                # If still too large, subsample faces more carefully
                if len(mesh.faces) > max_faces:
                    # Keep every Nth face to maintain some structure
                    step = len(mesh.faces) // max_faces
                    if step > 1:
                        keep_faces = np.arange(0, len(mesh.faces), step)[:max_faces]
                        mesh.update_faces(keep_faces)
                        # Clean up the mesh after face removal
                        mesh.remove_unreferenced_vertices()
                        
                print(f"    Used basic trimesh simplification")
                
            except Exception as e:
                print(f"    Simplification failed: {e}. Proceeding with original mesh.")
                # If all else fails, just warn and continue with original mesh
                print(f"    WARNING: Using full mesh - may require more memory")
        print(f"    Simplified: {original_faces} -> {len(mesh.faces)} faces ({len(mesh.vertices)} verts)")
    
    # Adjust grid size for low-memory mode
    if args.low_memory and args.grid_size > 64:
        args.grid_size = 64
        print(f"    Low-memory mode: reducing grid size to {args.grid_size}")

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

    # ── XRF: BFS region detection → flattest squares within best convex region ──
    print("\n[6/xrf] BFS flat-region detection...")
    best_region_vi = find_flat_regions(
        V_cm, mesh_cm.faces, face_areas, N_arr, H_arr, concave,
        curvature_percent=50.0,
        normal_similarity=0.50,
        min_region_vertex_count=30,
        target_sensor_area_sq_cm=2.0,
        curvature_tolerance_factor=10.0,
        concave_penalty_factor=50.0,
    )
    xrf_cand = best_region_vi if len(best_region_vi) else np.arange(len(V_cm))
    print(f"  Searching {len(xrf_cand)} candidate vertices for XRF squares...")
    xrf_results = find_xrf_squares(V_cm, xrf_cand, N_arr, H_local,
                                   concave=concave, n_points=3, concave_penalty=50.0,
                                   min_edge_distance_factor=args.edge_distance)
    for x in xrf_results:
        print(f"    {x['label']}  pt={x['point'].round(3)}"
              f"  half={x['half_size']:.3f}  h={x['h']:.4f}")

    # ── Raman: per-cluster connected-component targets ─────────────────────────
    if args.xrf_only:
        print("\n[6/raman] Skipped (--xrf-only)")
        raman_data = []
    else:
        print("\n[6/raman] Per-cluster Raman targets...")
        raman_data = []
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
            targets = find_raman_targets(V_cm, N_arr, mask_cube, best_M, face_buf, mesh_cm,
                                         n_points=3, min_points=args.min_crosshairs,
                                         grid=args.grid_size, valid_fi=fi_set)
            if not targets:
                print(f"    SKIP — fewer than {args.min_crosshairs} viable components")
                raman_data.append({"ci": ci, "color": color, "mask": mask_cube,
                                   "skipped": True, "reason": "insufficient components",
                                   "n_faces": int(len(fi)), "targets": []})
                continue

            print(f"    Raman targets: {len(targets)}")
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
                             "n_px": t.get("n_px")} for t in entry.get("targets", [])],
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
                     "h": round(float(x["h"]), 6)} for x in xrf_results],
    }
    xrf_rpt = out_dir / f"{art}_xrf_report.json"
    with open(xrf_rpt, "w") as f:
        json.dump(xrf_report, f, indent=2)
    print(f"    {xrf_rpt.name}")

    print(f"\nDone.\n  XRF:   {xrf_ply}"
          + (f"\n  Raman: {raman_ply}" if not args.xrf_only else "  (Raman unchanged)"))


if __name__ == "__main__":
    main()
