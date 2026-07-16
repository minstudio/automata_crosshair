"""
HSI-to-3D Flatness Registration — Proof of Concept

The HSI camera and photogrammetry rig are separate physical setups, so
texture-based matching (SIFT) fails. This script uses silhouette matching:

  1. Segment the object from the background in the HSI viewfinder (Otsu)
  2. Render binary silhouettes of the 3D mesh from many viewpoints
  3. Find the render+rotation whose silhouette best fits the viewfinder (IoU)
  4. Compute a similarity transform (scale + translation + rotation) from
     the silhouette alignment — warp the flatness render into HSI space
  5. Select the flattest point per HSI spectral cluster

Usage:
  python hsi/hsi_3d_registration.py \
      --mesh        path/to/mesh.obj \
      --viewfinder  path/to/viewfinder.png \
      --hdr         path/to/capture.hdr \
      --raw         path/to/capture.raw \
      [--clusters   path/to/_hdbscan_pca.png] \
      [--output     output_dir]
"""

import argparse
import json
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
import torch
import trimesh
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

warnings.filterwarnings("ignore")

# ── Pyrender offscreen setup ──────────────────────────────────────────────────
try:
    import pyrender
    try:
        _t = pyrender.OffscreenRenderer(16, 16); _t.delete()
    except Exception:
        import os; os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    HAS_PYRENDER = True
except ImportError:
    HAS_PYRENDER = False


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def cotangent_laplacian(vertices, faces):
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0

    def cot(a, b):
        cos = np.einsum("ij,ij->i", a, b)
        sin_sq = np.maximum(np.linalg.norm(np.cross(a, b), axis=1) ** 2, 1e-10)
        return np.clip(cos / np.sqrt(sin_sq), -1e5, 1e5)

    cot0, cot1, cot2 = cot(e1, e2), cot(e2, e0), cot(e0, e1)
    I = np.hstack([i1, i2, i2, i0, i0, i1])
    J = np.hstack([i2, i1, i0, i2, i1, i0])
    W = 0.5 * np.hstack([cot0, cot0, cot1, cot1, cot2, cot2])
    L = sp.coo_matrix((W, (I, J)), shape=(len(vertices), len(vertices)))
    diag = -np.asarray(L.sum(axis=1)).ravel()
    L = L + sp.diags(diag)
    return 0.5 * (L + L.T).tocsr()


def compute_vertex_areas(vertices, faces, face_areas):
    va = np.zeros(len(vertices))
    np.add.at(va, faces[:, 0], face_areas / 3.0)
    np.add.at(va, faces[:, 1], face_areas / 3.0)
    np.add.at(va, faces[:, 2], face_areas / 3.0)
    return va


def compute_flatness_scores(mesh):
    """Per-vertex flatness: 1/(curvature+eps), normalised to [0,1]. High=flat."""
    vertices = mesh.vertices.copy()
    faces = mesh.faces
    vertices -= vertices.mean(axis=0)
    scale = np.linalg.norm(vertices, axis=1).max()
    if scale > 0:
        vertices /= scale

    face_areas = mesh.area_faces
    vertex_areas = compute_vertex_areas(vertices, faces, face_areas)
    L = cotangent_laplacian(vertices, faces)
    Lx = L @ vertices
    Lx /= np.maximum(vertex_areas, 1e-20)[:, None]
    curvature = 0.5 * np.linalg.norm(Lx, axis=1)

    flatness = 1.0 / (curvature + 1e-6)
    f_min, f_max = flatness.min(), flatness.max()
    if f_max > f_min:
        flatness = (flatness - f_min) / (f_max - f_min)
    else:
        flatness[:] = 0.5
    return flatness


# ─────────────────────────────────────────────────────────────────────────────
# HSI loading
# ─────────────────────────────────────────────────────────────────────────────

def load_envi_raw(hdr_path, raw_path):
    """ENVI BIL → H×W×B float32."""
    with open(hdr_path, "r") as f:
        content = f.read()
    hdr = {}
    for line in content.splitlines():
        if "=" in line:
            k, v = line.split("=", 1); hdr[k.strip().lower()] = v.strip()
    samples = int(hdr.get("samples", 512))
    lines_  = int(hdr.get("lines",   512))
    bands_  = int(hdr.get("bands",   204))
    dtype   = {1: np.uint8, 2: np.int16, 12: np.uint16}.get(
                  int(hdr.get("data type", 12)), np.uint16)
    data = np.fromfile(raw_path, dtype=dtype).reshape(lines_, bands_, samples).astype(np.float32)
    return data.transpose(0, 2, 1)


def parse_default_bands(hdr_path):
    with open(hdr_path, "r") as f:
        content = f.read()
    m = re.search(r"default bands\s*=\s*\{([^}]+)\}", content, re.IGNORECASE)
    return [int(x.strip()) for x in m.group(1).split(",")] if m else [70, 53, 19]


def parse_datacube_angle(xml_path):
    if xml_path is None or not Path(xml_path).exists():
        return 0
    try:
        tree = ET.parse(str(xml_path))
        for elem in tree.iter("key"):
            if elem.get("field") == "datacube_angle":
                return int(elem.text.strip())
    except Exception:
        pass
    return 0


def load_viewfinder(viewfinder_path, hdr_path=None, raw_path=None, xml_path=None):
    """Load viewfinder PNG, apply datacube_angle rotation."""
    img = None
    if viewfinder_path and Path(viewfinder_path).exists():
        img = cv2.imread(str(viewfinder_path))

    if img is None or img.mean() < 5:
        print("  Viewfinder dark/missing — building pseudo-RGB from cube")
        cube = load_envi_raw(hdr_path, raw_path)
        bands = parse_default_bands(hdr_path)
        r, g, b = cube[:, :, bands[0]], cube[:, :, bands[1]], cube[:, :, bands[2]]
        pseudo = np.stack([b, g, r], axis=2)
        top = pseudo.max()
        img = ((pseudo / top * 255) if top > 0 else pseudo).astype(np.uint8)

    angle = parse_datacube_angle(xml_path)
    if angle != 0:
        k = ((-angle) // 90) % 4
        print(f"  datacube_angle={angle} deg, rot90 k={k}")
        img = np.rot90(img, k=k)
        if img.shape[0] != img.shape[1]:
            img = cv2.resize(img, (512, 512))
    return img.copy()  # ensure C-contiguous


# ─────────────────────────────────────────────────────────────────────────────
# Viewfinder segmentation — BiRefNet
# ─────────────────────────────────────────────────────────────────────────────

_BIREFNET_MODEL  = None
_BIREFNET_DEVICE = None
_BIREFNET_ID     = "ZhengPeng7/BiRefNet"

_BIREFNET_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _get_birefnet():
    global _BIREFNET_MODEL, _BIREFNET_DEVICE
    if _BIREFNET_MODEL is None:
        print(f"  Loading BiRefNet ({_BIREFNET_ID})...")
        _BIREFNET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _BIREFNET_MODEL = AutoModelForImageSegmentation.from_pretrained(
            _BIREFNET_ID, trust_remote_code=True)
        _BIREFNET_MODEL.to(_BIREFNET_DEVICE)
        _BIREFNET_MODEL.eval()
        print(f"  BiRefNet loaded on {_BIREFNET_DEVICE}")
    return _BIREFNET_MODEL, _BIREFNET_DEVICE


def segment_viewfinder(viewfinder_bgr):
    """
    Segment artifact from background using BiRefNet + morphological cleanup.
    Returns binary mask (uint8, 255=object).
    """
    h, w = viewfinder_bgr.shape[:2]

    # BGR (OpenCV) → RGB PIL for BiRefNet
    rgb = cv2.cvtColor(viewfinder_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    model, device = _get_birefnet()
    input_tensor = _BIREFNET_TRANSFORM(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)
        if isinstance(preds, (list, tuple)):
            preds = preds[-1]
        preds = torch.sigmoid(preds)
        preds = torch.nn.functional.interpolate(
            preds, size=(h, w), mode="bilinear", align_corners=False)
        prob = preds.squeeze().cpu().numpy()

    binary = (prob > 0.5).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)

    # Keep only the largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = (labels == largest).astype(np.uint8) * 255

    coverage = (binary > 0).mean() * 100
    print(f"  Viewfinder object coverage: {coverage:.1f}%")
    return binary


def silhouette_from_render(clay_img_rgb):
    """Binary silhouette from a clay render (non-black pixels)."""
    gray = cv2.cvtColor(clay_img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return binary


# ─────────────────────────────────────────────────────────────────────────────
# Silhouette matching
# ─────────────────────────────────────────────────────────────────────────────

def silhouette_moments(sil):
    M = cv2.moments(sil)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    area = M["m00"]
    # Angle from second order moments
    if abs(M["mu20"] - M["mu02"]) < 1e-10 and abs(M["mu11"]) < 1e-10:
        angle = 0.0
    else:
        angle = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
    return {"cx": cx, "cy": cy, "area": area, "angle": angle}


def _similarity_transform(render_sil, r_mom, vf_mom, theta, h, w):
    """
    Similarity transform (rotation θ + uniform scale + translation) that maps
    the render silhouette centroid/scale onto the VF silhouette.
    Returns (warped_bin, M_3x3).
    """
    scale = np.sqrt(vf_mom["area"] / (r_mom["area"] + 1e-8))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx_r, cy_r = r_mom["cx"], r_mom["cy"]
    cx_v, cy_v = vf_mom["cx"], vf_mom["cy"]

    M = np.float32([
        [scale * cos_t, -scale * sin_t,
         cx_v - scale * (cos_t * cx_r - sin_t * cy_r)],
        [scale * sin_t,  scale * cos_t,
         cy_v - scale * (sin_t * cx_r + cos_t * cy_r)],
    ])
    warped = cv2.warpAffine(render_sil.astype(np.float32), M, (w, h))
    warped_bin = (warped > 127).astype(np.uint8) * 255
    M_3x3 = np.eye(3, dtype=np.float64)
    M_3x3[:2, :] = M.astype(np.float64)
    return warped_bin, M_3x3


def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(inter) / (float(union) + 1e-8)


def find_best_render_by_silhouette(vf_mask, renders):
    """
    Align each render silhouette to the VF mask using a full similarity
    transform (rotation + scale + translation).

    Strategy:
      1. Coarse pass — principal-axis angle ± 180° ambiguity for every render.
      2. Fine pass   — ±30° sweep at 3° steps for the top-5 coarse candidates.

    Returns (best_idx, 0, best_iou, best_M_3x3, best_render_sil).
    """
    h, w = vf_mask.shape

    vf_mom = silhouette_moments(vf_mask)
    if vf_mom is None:
        print("  WARNING: no foreground in viewfinder mask")
        return 0, 0, 0.0, np.eye(3), None

    # ── coarse pass ───────────────────────────────────────────────────────
    coarse = []   # (iou, render_idx, theta, warped, M, render_sil, r_mom)
    for i, r in enumerate(renders):
        render_sil = silhouette_from_render(r["clay_img"])
        r_mom = silhouette_moments(render_sil)
        if r_mom is None or r_mom["area"] < 100:
            continue
        base = vf_mom["angle"] - r_mom["angle"]
        for extra in (0.0, np.pi):
            theta = base + extra
            warped, M = _similarity_transform(render_sil, r_mom, vf_mom, theta, h, w)
            score = iou(vf_mask, warped)
            coarse.append((score, i, theta, warped, M, render_sil, r_mom))

    if not coarse:
        return 0, 0, 0.0, np.eye(3), None

    coarse.sort(key=lambda x: -x[0])
    best_score, best_idx, best_theta, best_warped, best_M, _, _ = coarse[0]

    # ── fine pass on top-5 candidates ─────────────────────────────────────
    fine_angles = np.deg2rad(np.arange(-30, 31, 3))
    seen = set()
    for entry in coarse[:5]:
        _, i, coarse_theta, _, _, render_sil, r_mom = entry
        if i in seen:
            continue
        seen.add(i)
        for delta in fine_angles:
            warped, M = _similarity_transform(
                render_sil, r_mom, vf_mom, coarse_theta + delta, h, w)
            score = iou(vf_mask, warped)
            if score > best_score:
                best_score  = score
                best_idx    = i
                best_theta  = coarse_theta + delta
                best_warped = warped
                best_M      = M

    print(f"  Best render={best_idx}, angle={np.rad2deg(best_theta):.1f} deg, "
          f"silhouette IoU={best_score:.3f}")
    return best_idx, 0, best_score, best_M, best_warped


# ─────────────────────────────────────────────────────────────────────────────
# Rendering (clay + flatness)
# ─────────────────────────────────────────────────────────────────────────────

def look_at_pose(eye, center=np.zeros(3), up=np.array([0, 1, 0])):
    eye = np.asarray(eye, dtype=float)
    z = eye - center; z /= np.linalg.norm(z) + 1e-8
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(np.array([1., 0., 0.]), z)
    x /= np.linalg.norm(x) + 1e-8
    y = np.cross(z, x)
    pose = np.eye(4); pose[:3, 0] = x; pose[:3, 1] = y; pose[:3, 2] = z; pose[:3, 3] = eye
    return pose


def fibonacci_viewpoints(radius=2.0):
    """19-20 viewpoints: fibonacci sphere + mandatory cardinals."""
    golden = (1 + np.sqrt(5)) / 2
    pts = []
    for i in range(14):
        theta = np.arccos(1 - 2 * (i + 0.5) / 14)
        phi   = 2 * np.pi * i / golden
        pts.append((radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(theta)))
    cardinals = [(0,0,radius),(0,0,-radius),(radius,0,0),(-radius,0,0),(0,radius,0),(0,-radius,0)]
    unique = list(cardinals)
    for p in pts:
        if not any(np.linalg.norm(np.array(p)-np.array(c)) < 0.3*radius for c in cardinals):
            unique.append(p)
    return unique[:20]


def render_clay_views(tri_mesh_norm, flatness, viewpoints, size=512):
    if not HAS_PYRENDER:
        print("  pyrender not available — no renders")
        return []
    renderer = pyrender.OffscreenRenderer(size, size)
    results = []
    for eye in viewpoints:
        pose = look_at_pose(eye)

        # Clay render
        clay = tri_mesh_norm.copy()
        clay.visual = trimesh.visual.ColorVisuals(
            mesh=clay, vertex_colors=np.tile([200, 200, 200, 255], (len(clay.vertices), 1)))
        sc = pyrender.Scene(bg_color=[0,0,0,255], ambient_light=[0.6,0.6,0.6])
        sc.add(pyrender.Mesh.from_trimesh(clay, smooth=True))
        sc.add(pyrender.PerspectiveCamera(yfov=np.pi/3, znear=0.01, zfar=100), pose=pose)
        sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=6.0), pose=pose)
        sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
               pose=look_at_pose(-np.array(eye)*1.5))
        clay_img, _ = renderer.render(sc)

        # Flatness render
        colors = np.zeros((len(tri_mesh_norm.vertices), 4), dtype=np.uint8)
        colors[:, :3] = (flatness[:, None] * 255).astype(np.uint8)
        colors[:, 3]  = 255
        flat_mesh = tri_mesh_norm.copy()
        flat_mesh.visual = trimesh.visual.ColorVisuals(mesh=flat_mesh, vertex_colors=colors)
        sc2 = pyrender.Scene(bg_color=[0,0,0,255], ambient_light=[1.0,1.0,1.0])
        sc2.add(pyrender.Mesh.from_trimesh(flat_mesh, smooth=True))
        sc2.add(pyrender.PerspectiveCamera(yfov=np.pi/3, znear=0.01, zfar=100), pose=pose)
        flat_img, _ = renderer.render(sc2)

        results.append({"eye": eye, "clay_img": clay_img, "flat_img": flat_img})
    renderer.delete()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Cluster target selection
# ─────────────────────────────────────────────────────────────────────────────

MIN_CLUSTER_AREA_PX = 20


def find_best_point(mask, flatness_map=None):
    if flatness_map is not None and np.any(flatness_map[mask] > 0):
        masked = np.where(mask, flatness_map, -np.inf)
        idx = np.argmax(masked)
        row, col = np.unravel_index(idx, mask.shape)
        return row, col, int(mask.sum())

    labeled, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    best_label = int(np.argmax(sizes)) + 1
    area = int(sizes[best_label - 1])
    if area < MIN_CLUSTER_AREA_PX:
        return None
    cy, cx = ndimage.center_of_mass(labeled == best_label)
    cy_int, cx_int = int(round(cy)), int(round(cx))
    lm = labeled == best_label
    if not lm[cy_int, cx_int]:
        ys, xs = np.where(lm)
        nearest = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
        cy_int, cx_int = ys[nearest], xs[nearest]
    return cy_int, cx_int, area


def select_targets_from_clusters(clusters_path, flatness_hsi):
    img = cv2.imread(str(clusters_path))
    arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = arr.reshape(-1, 3)
    non_black = pixels[np.any(pixels > 10, axis=1)]
    if len(non_black) == 0:
        return []
    targets = []
    for color in np.unique(non_black, axis=0):
        mask = np.all(arr == color, axis=2)
        result = find_best_point(mask, flatness_hsi)
        if result is None:
            continue
        row, col, area = result
        flat_val = float(flatness_hsi[row, col]) if flatness_hsi is not None else None
        targets.append({
            "color_rgb": [int(c) for c in color],
            "pixel_x": int(col), "pixel_y": int(row),
            "region_area_px": area,
            "flatness_score": flat_val,
        })
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# Debug visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_debug_image(output_dir, viewfinder_bgr, vf_mask, best_clay_rgb,
                     aligned_sil, flatness_hsi, iou_score, label=""):
    """4-panel: viewfinder | VF mask | best render | flatness map."""
    size = 512
    vf   = cv2.resize(viewfinder_bgr, (size, size))
    msk  = cv2.cvtColor(cv2.resize(vf_mask, (size, size)), cv2.COLOR_GRAY2BGR)
    clay = cv2.cvtColor(cv2.resize(best_clay_rgb, (size, size)), cv2.COLOR_RGB2BGR)

    # Draw the warped render silhouette (VF space) on the VF mask panel —
    # both are in the same coordinate system so overlap = registration quality.
    if aligned_sil is not None:
        contours, _ = cv2.findContours(cv2.resize(aligned_sil, (size, size)),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(msk, contours, -1, (0, 255, 0), 2)

    if flatness_hsi is not None:
        flat_vis = cv2.applyColorMap(
            (flatness_hsi * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        flat_vis = cv2.resize(flat_vis, (size, size))
    else:
        flat_vis = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.putText(flat_vis, "no flatness map", (40, 256),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)

    labels = ["Viewfinder", "VF silhouette", f"Best render (IoU={iou_score:.2f})", "Flatness (HSI space)"]
    for img, text in zip([vf, msk, clay, flat_vis], labels):
        cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1)

    out = Path(output_dir) / f"registration_debug{label}.png"
    cv2.imwrite(str(out), np.hstack([vf, msk, clay, flat_vis]))
    print(f"  Debug image: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_registration(mesh_path, viewfinder_path, hdr_path, raw_path,
                     clusters_path, output_dir, render_size=512, mask_path=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: HSI viewfinder ────────────────────────────────────────────────
    print("\n[1/6] Loading HSI viewfinder...")
    xml_path = None
    if hdr_path:
        cand = Path(hdr_path).parent.parent / "metadata" / f"{Path(hdr_path).stem}.xml"
        if cand.exists():
            xml_path = cand
            print(f"  Metadata XML: {cand.name}")
    viewfinder = load_viewfinder(viewfinder_path, hdr_path, raw_path, xml_path)
    viewfinder = cv2.resize(viewfinder, (render_size, render_size))
    print(f"  Shape: {viewfinder.shape}, mean: {viewfinder.mean():.1f}")

    # Segment object from background
    if mask_path and Path(mask_path).exists():
        print(f"  Using pre-computed mask: {Path(mask_path).name}")
        vf_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # apply the same datacube_angle rotation as the viewfinder
        angle = parse_datacube_angle(xml_path)
        if angle != 0:
            k = ((-angle) // 90) % 4
            vf_mask = np.rot90(vf_mask, k=k).copy()
        vf_mask = cv2.resize(vf_mask, (render_size, render_size),
                             interpolation=cv2.INTER_NEAREST)
        _, vf_mask = cv2.threshold(vf_mask, 127, 255, cv2.THRESH_BINARY)
        coverage = (vf_mask > 0).mean() * 100
        print(f"  Object coverage: {coverage:.1f}%")
    else:
        if mask_path:
            print(f"  Mask not found at {mask_path}, falling back to BiRefNet")
        vf_mask = segment_viewfinder(viewfinder)

    # ── Step 2: Mesh + flatness ───────────────────────────────────────────────
    print("\n[2/6] Loading mesh and computing flatness scores...")
    mesh = trimesh.load(str(mesh_path), force="mesh")
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    flatness = compute_flatness_scores(mesh)
    print(f"  Flatness range: [{flatness.min():.3f}, {flatness.max():.3f}]")

    # ── Step 3: Normalise mesh for rendering ──────────────────────────────────
    print("\n[3/6] Generating candidate renders...")
    mesh_norm = mesh.copy()
    bbox_min, bbox_max = mesh_norm.bounds
    center = (bbox_min + bbox_max) / 2.0
    half_ext = np.max(bbox_max - bbox_min) / 2.0
    if half_ext > 0:
        mesh_norm.vertices = (mesh_norm.vertices - center) / half_ext

    viewpoints = fibonacci_viewpoints(radius=2.0)
    renders = render_clay_views(mesh_norm, flatness, viewpoints, size=render_size)
    print(f"  Rendered {len(renders)} views")

    renders_dir = output_dir / "candidate_renders"
    renders_dir.mkdir(exist_ok=True)
    for i, r in enumerate(renders):
        cv2.imwrite(str(renders_dir / f"clay_{i:02d}.png"),
                    cv2.cvtColor(r["clay_img"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(renders_dir / f"flat_{i:02d}.png"),
                    cv2.cvtColor(r["flat_img"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "vf_silhouette.png"), vf_mask)

    # ── Step 4: Silhouette-based viewpoint matching ───────────────────────────
    print("\n[4/6] Silhouette matching (render × 4 rotations)...")
    best_idx, best_rot_k, best_iou_val, best_M, aligned_sil = \
        find_best_render_by_silhouette(vf_mask, renders)

    best_render = renders[best_idx]

    # ── Step 5: Warp flatness render into HSI space ───────────────────────────
    print("\n[5/6] Warping flatness map into HSI pixel space...")

    # Warp flatness render → viewfinder space using the similarity transform
    flat_gray = cv2.cvtColor(best_render["flat_img"], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    flatness_hsi = cv2.warpAffine(
        flat_gray, best_M[:2, :], (render_size, render_size),
        flags=cv2.INTER_LINEAR, borderValue=0.0)

    flatness_hsi = np.clip(flatness_hsi, 0, 1)
    np.save(str(output_dir / "flatness_map_hsi.npy"), flatness_hsi.astype(np.float32))

    coverage = (flatness_hsi > 0.01).mean() * 100
    print(f"  Flatness map coverage in HSI space: {coverage:.1f}%")

    # Confidence based on IoU quality
    if best_iou_val >= 0.4:
        registration_confidence = "medium"
    elif best_iou_val >= 0.2:
        registration_confidence = "low"
    else:
        registration_confidence = "none"
        flatness_hsi = None

    print(f"  IoU={best_iou_val:.3f} -> confidence: {registration_confidence}")

    # ── Step 6: Cluster targets ───────────────────────────────────────────────
    print("\n[6/6] Selecting measurement targets...")
    targets = []
    if clusters_path and Path(clusters_path).exists():
        targets = select_targets_from_clusters(clusters_path, flatness_hsi)
        print(f"  {len(targets)} targets selected")
        for t in targets:
            fs = t.get("flatness_score")
            print(f"    ({t['pixel_x']}, {t['pixel_y']}) flatness={fs:.3f}" if fs else
                  f"    ({t['pixel_x']}, {t['pixel_y']})")
    else:
        print("  No cluster map provided")

    result = {
        "mesh": str(mesh_path),
        "viewfinder": str(viewfinder_path),
        "best_render_idx": best_idx,
        "best_render_eye": list(best_render["eye"]),
        "best_rotation_deg": float(np.rad2deg(best_rot_k)),
        "silhouette_iou": float(best_iou_val),
        "registration_confidence": registration_confidence,
        "targets": targets,
    }
    with open(output_dir / "raman_targets_flat.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Output JSON: {output_dir / 'raman_targets_flat.json'}")

    save_debug_image(output_dir, viewfinder, vf_mask,
                     best_render["clay_img"], aligned_sil, flatness_hsi, best_iou_val)

    print(f"\nRegistration confidence: {registration_confidence.upper()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="HSI-to-3D flatness registration PoC")
    p.add_argument("--mesh",       required=True)
    p.add_argument("--viewfinder", default=None)
    p.add_argument("--hdr",        default=None)
    p.add_argument("--raw",        default=None)
    p.add_argument("--clusters",   default=None)
    p.add_argument("--output",     default="output_registration")
    p.add_argument("--mask",       default=None,
                   help="Pre-computed silhouette mask PNG (from segment_roi_birefnet.py). "
                        "If omitted, auto-detected from hsi/output/roi/<artifact>_mask.png.")
    args = p.parse_args()

    # Auto-detect mask from roi output if not explicitly given
    mask_path = args.mask
    if mask_path is None:
        artifact_name = Path(args.output).name
        auto = Path(__file__).parent / "output" / "roi" / f"{artifact_name}_mask.png"
        if auto.exists():
            print(f"  Auto-detected mask: {auto}")
            mask_path = str(auto)

    result = run_registration(
        mesh_path=args.mesh,
        viewfinder_path=args.viewfinder,
        hdr_path=args.hdr,
        raw_path=args.raw,
        clusters_path=args.clusters,
        output_dir=args.output,
        mask_path=mask_path,
    )
    print(f"\nDone. Confidence: {result['registration_confidence']}")


if __name__ == "__main__":
    main()
