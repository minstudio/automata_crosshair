"""
Test: PCA-driven viewpoint selection vs fibonacci sphere.

Instead of 19 random viewpoints, generate ~10 targeted views by finding
the mesh's thin axis (3rd PCA component) — the "lying flat on table" view.

Usage:
    python hsi/test_pca_viewpoints.py \
        --mesh path/to/mesh.obj \
        --mask path/to/mask.png \
        --hdr  path/to/capture.hdr \
        [--viewfinder path/to/viewfinder.png] \
        [--output test_pca_out]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hsi_3d_registration import (
    compute_flatness_scores,
    find_best_render_by_silhouette,
    load_viewfinder,
    parse_datacube_angle,
    render_clay_views,
    save_debug_image,
    silhouette_from_render,
)


def pca_viewpoints(mesh_norm, radius=2.0, tilt_angles_deg=(0, 15, 30)):
    """
    Generate viewpoints concentrated around the mesh's thin axis.

    The 3rd PCA axis (smallest variance) is the normal to the flat face —
    looking along it shows maximum projected area (the 'lying on table' view).

    Returns a list of (x, y, z) eye positions at the given radius.
    """
    verts    = mesh_norm.vertices
    centered = verts - verts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    thin_axis = Vt[2]
    mid_axis  = Vt[1]
    maj_axis  = Vt[0]

    print(f"  PCA thin axis: {thin_axis.round(3)}")

    viewpoints = []
    seen = set()

    def add(eye):
        eye = np.asarray(eye, dtype=float)
        n   = np.linalg.norm(eye)
        if n < 1e-8:
            return
        key = tuple((eye / n).round(2))
        if key not in seen:
            seen.add(key)
            viewpoints.append((eye / n * radius).tolist())

    for tilt_deg in tilt_angles_deg:
        tilt = np.deg2rad(tilt_deg)
        if tilt_deg == 0:
            add( thin_axis)
            add(-thin_axis)
        else:
            for az_deg in (0, 90, 180, 270):
                az  = np.deg2rad(az_deg)
                perp = np.cos(az) * maj_axis + np.sin(az) * mid_axis
                eye  = np.cos(tilt) * thin_axis + np.sin(tilt) * perp
                add( eye)
                add(-eye)

    print(f"  Generated {len(viewpoints)} PCA-driven viewpoints")
    return viewpoints


def projected_area(render):
    sil = silhouette_from_render(render["clay_img"])
    return (sil > 0).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh",        required=True)
    ap.add_argument("--mask",        required=True,
                    help="Pre-computed BiRefNet mask PNG")
    ap.add_argument("--hdr",         default=None)
    ap.add_argument("--raw",         default=None)
    ap.add_argument("--viewfinder",  default=None)
    ap.add_argument("--output",      default="test_pca_out")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    size = 512

    print("\n[1] Loading viewfinder...")
    xml_path = None
    if args.hdr:
        cand = Path(args.hdr).parent.parent / "metadata" / f"{Path(args.hdr).stem}.xml"
        if cand.exists():
            xml_path = cand
    viewfinder = load_viewfinder(args.viewfinder, args.hdr, args.raw, xml_path)
    viewfinder = cv2.resize(viewfinder, (size, size))

    print("[2] Loading mask...")
    vf_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    angle = parse_datacube_angle(xml_path)
    if angle != 0:
        k = ((-angle) // 90) % 4
        vf_mask = np.rot90(vf_mask, k=k).copy()
        print(f"  Applied datacube_angle rotation k={k}")
    vf_mask = cv2.resize(vf_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    _, vf_mask = cv2.threshold(vf_mask, 127, 255, cv2.THRESH_BINARY)
    print(f"  Coverage: {(vf_mask > 0).mean()*100:.1f}%")

    print("[3] Loading mesh...")
    mesh = trimesh.load(str(args.mesh), force="mesh")
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    mesh_norm = mesh.copy()
    bbox_min, bbox_max = mesh_norm.bounds
    center   = (bbox_min + bbox_max) / 2.0
    half_ext = np.max(bbox_max - bbox_min) / 2.0
    if half_ext > 0:
        mesh_norm.vertices = (mesh_norm.vertices - center) / half_ext

    flatness = compute_flatness_scores(mesh)

    print("\n[4] PCA-driven viewpoints...")
    viewpoints = pca_viewpoints(mesh_norm)
    renders    = render_clay_views(mesh_norm, flatness, viewpoints, size=size)
    print(f"  Rendered {len(renders)} views")

    renders_dir = out / "pca_renders"
    renders_dir.mkdir(exist_ok=True)
    areas = []
    for i, r in enumerate(renders):
        cv2.imwrite(str(renders_dir / f"clay_{i:02d}.png"),
                    cv2.cvtColor(r["clay_img"], cv2.COLOR_RGB2BGR))
        area = projected_area(r)
        areas.append(area)
        print(f"  render {i:02d}  eye={[round(x,2) for x in r['eye']]}  "
              f"projected_area={area}px")

    best_by_area = int(np.argmax(areas))
    print(f"\n  Largest projected area: render {best_by_area} "
          f"({areas[best_by_area]}px)")

    print("\n[5] Silhouette matching...")
    best_idx, _, best_iou, best_M, aligned_sil = \
        find_best_render_by_silhouette(vf_mask, renders)

    flat_gray = cv2.cvtColor(
        renders[best_idx]["flat_img"], cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0
    flatness_hsi = cv2.warpAffine(
        flat_gray, best_M[:2, :], (size, size),
        flags=cv2.INTER_LINEAR, borderValue=0.0)
    flatness_hsi = np.clip(flatness_hsi, 0, 1)
    np.save(str(out / "flatness_map_hsi.npy"), flatness_hsi.astype(np.float32))

    coverage = (flatness_hsi > 0.01).mean() * 100
    print(f"  Flatness coverage: {coverage:.1f}%")

    save_debug_image(out, viewfinder, vf_mask,
                     renders[best_idx]["clay_img"], aligned_sil,
                     flatness_hsi, best_iou, label="_pca")

    print(f"\nIoU: {best_iou:.3f}  (PCA-driven, {len(renders)} views)")
    print(f"Debug image: {out / 'registration_debug_pca.png'}")


if __name__ == "__main__":
    main()
