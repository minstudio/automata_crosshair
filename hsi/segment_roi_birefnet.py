"""
BiRefNet segmentation of the artifact in HSI viewfinder/reflectance images.

Default (automatic, no user input):
    1. BiRefNet on the full image.
    2. Keep the largest foreground component that does not touch the image
       border (the platform ring / fixtures do), fill holes.
    3. Crop around it with padding and run BiRefNet again on the crop, so the
       model sees the object up close; keep the component overlapping the
       first-pass mask.

--manual: the old interactive mode, where you draw the ROI box yourself
(needed for cluttered scenes).

Usage — single image:
    python hsi/segment_roi_birefnet.py --viewfinder path/to/viewfinder.png --output path/to/mask.png

Usage — batch over all artifact folders that have HSI:
    python hsi/segment_roi_birefnet.py --batch --hsi_only --output_dir hsi/output/roi

Manual mode controls (per image):
    Left-drag   draw / redraw the ROI rectangle
    Enter/R     run BiRefNet on current ROI
    S           save mask and move to next image
    D           discard / skip this image (no file written)
    Q           quit immediately
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gpu

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from paths import DATASET_ROOT

# ── BiRefNet ──────────────────────────────────────────────────────────────────

_MODEL_ID = "ZhengPeng7/BiRefNet"
_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
_model  = None
_device = None


def _load_model():
    global _model, _device
    if _model is None:
        print(f"Loading BiRefNet ({_MODEL_ID})...")
        _model, _device = gpu.load_birefnet(_MODEL_ID)
    return _model, _device


def run_birefnet_on_crop(bgr_crop):
    """Run BiRefNet on a BGR crop. Returns float32 prob map at crop size."""
    h, w = bgr_crop.shape[:2]
    rgb   = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb)
    model, device = _load_model()
    tensor = _TRANSFORM(pil).unsqueeze(0)
    preds = gpu.birefnet_forward(model, tensor)
    with torch.no_grad():
        preds = torch.nn.functional.interpolate(
            preds, size=(h, w), mode="bilinear", align_corners=False)
    return preds.squeeze().cpu().numpy()   # float32 [0,1]


# ── Automatic segmentation ──────────────────────────────────────────────────────

def _components(mask_u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    return n, labels, stats


def _fill_holes(mask_u8):
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return mask_u8 | cv2.bitwise_not(flood)


def _pick_object(prob, threshold=0.5, border=2, min_frac=0.003):
    """Pick the artifact from a foreground probability map.

    The largest component that does not touch the image border wins (the
    platform ring and fixtures do touch it). If the artifact is joined to
    them by thin bridges, increasingly strong openings are tried to cut the
    bridges; the chosen piece is then grown back inside the original mask.
    Falls back to the largest component overall. Returns uint8 0/255 or None."""
    m = (prob > threshold).astype(np.uint8) * 255
    if not m.any():
        return None
    H, W = m.shape
    min_area = int(min_frac * H * W)
    for k in (0, 5, 9, 15):
        mk = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) if k else m
        n, labels, stats = _components(mk)
        best, best_area = None, 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            touches = x <= border or y <= border or x + w >= W - border or y + h >= H - border
            if not touches and area >= min_area and area > best_area:
                best, best_area = i, area
        if best is not None:
            piece = ((labels == best) * 255).astype(np.uint8)
            if k:
                grown = cv2.dilate(piece, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 2, k + 2)))
                piece = cv2.bitwise_and(grown, m)
                n2, l2, s2 = _components(piece)
                if n2 > 2:
                    piece = ((l2 == 1 + int(np.argmax(s2[1:, cv2.CC_STAT_AREA]))) * 255).astype(np.uint8)
                print(f"  separated the artifact from border-touching structures (opening {k}px)")
            return _fill_holes(piece)
    n, labels, stats = _components(m)
    pick = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    print("  [!] no artifact-sized component away from the image border; using the largest")
    return _fill_holes(((labels == pick) * 255).astype(np.uint8))


def auto_segment(bgr, pad_frac=0.25, threshold=0.5):
    """Automatic two-pass BiRefNet segmentation. Returns (mask uint8 0/255, info) or (None, reason)."""
    H, W = bgr.shape[:2]
    prob1 = run_birefnet_on_crop(bgr)
    m1 = _pick_object(prob1, threshold)
    if m1 is None:
        return None, "no foreground found"
    ys, xs = np.where(m1 > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int((y1 - y0) * pad_frac) + 8, int((x1 - x0) * pad_frac) + 8
    y0, y1 = max(0, y0 - py), min(H, y1 + py)
    x0, x1 = max(0, x0 - px), min(W, x1 + px)
    prob2 = np.zeros((H, W), np.float32)
    prob2[y0:y1, x0:x1] = run_birefnet_on_crop(bgr[y0:y1, x0:x1])
    m2 = (prob2 > threshold).astype(np.uint8) * 255
    n, labels, stats = _components(m2)
    if n > 1:
        overlap = [int(((labels == i) & (m1 > 0)).sum()) for i in range(1, n)]
        k = int(np.argmax(overlap)) + 1
        if overlap[k - 1] > 0:
            m2 = _fill_holes(((labels == k) * 255).astype(np.uint8))
        else:
            m2 = m1
    else:
        m2 = m1
    frac = (m2 > 0).mean()
    info = f"crop {x1 - x0}x{y1 - y0}, mask {int((m2 > 0).sum())} px ({100 * frac:.1f}%)"
    if frac < 0.002 or frac > 0.6:
        info += "  [!] unusual mask size, check the overlay (or rerun with --manual)"
    return m2, info


def save_mask(viewfinder_bgr, mask, name, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{name}_mask.png"
    cv2.imwrite(str(mask_path), mask)
    overlay = viewfinder_bgr.copy()
    fg = mask > 127
    overlay[fg] = (overlay[fg] * 0.45 + np.array([0, 220, 0]) * 0.55).astype(np.uint8)
    cts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cts, -1, (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / f"{name}_mask_overlay.png"), overlay)
    print(f"  Saved: {mask_path}")
    return mask_path


def process_one_auto(viewfinder_bgr, name, output_dir):
    print(f"\n  {name}: automatic segmentation")
    mask, info = auto_segment(viewfinder_bgr)
    if mask is None:
        print(f"  FAILED: {info}")
        return "skip"
    print(f"  {info}")
    save_mask(viewfinder_bgr, mask, name, output_dir)
    return "save"


# ── HSI I/O ───────────────────────────────────────────────────────────────────

def find_viewfinder(artifact_path):
    """Return (viewfinder_bgr, name) for an artifact folder, or (None, name)."""
    p = Path(artifact_path)
    name = p.name
    raw_data = p / "HSI" / "raw_data"
    if not raw_data.exists():
        return None, name

    # prefer the reflectance PNG from the results folder
    results = raw_data / "results"
    if results.exists():
        ref_pngs = sorted(results.glob("REFLECTANCE_*.png"))
        if ref_pngs:
            img = cv2.imread(str(ref_pngs[0]))
            if img is not None and img.mean() >= 5:
                return img, name

    # fallback: pseudo-RGB from the .raw cube
    capture = raw_data / "capture"
    if capture.exists():
        files  = list(capture.iterdir())
        raw_f  = next((f for f in files if f.suffix == ".raw"
                       and "DARK" not in f.name and "WHITE" not in f.name), None)
        hdr_f  = next((f for f in files if f.suffix == ".hdr"
                       and "DARK" not in f.name and "WHITE" not in f.name), None)
        if raw_f and hdr_f:
            bgr = _pseudo_rgb_from_cube(hdr_f, raw_f)
            if bgr is not None:
                return bgr, name
    return None, name


def _pseudo_rgb_from_cube(hdr_path, raw_path):
    try:
        with open(hdr_path) as f:
            content = f.read()
        hdr = {}
        for line in content.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                hdr[k.strip().lower()] = v.strip()
        samples   = int(hdr.get("samples", 512))
        lines_    = int(hdr.get("lines",   512))
        bands_    = int(hdr.get("bands",   204))
        dtype_map = {1: np.uint8, 2: np.int16, 12: np.uint16}
        dtype     = dtype_map.get(int(hdr.get("data type", 12)), np.uint16)
        data      = np.fromfile(raw_path, dtype=dtype)
        cube      = data.reshape(lines_, bands_, samples).transpose(0, 2, 1).astype(np.float32)
        r = cube[:, :, int(bands_ * 0.70)]
        g = cube[:, :, int(bands_ * 0.50)]
        b = cube[:, :, int(bands_ * 0.20)]
        rgb = np.stack([r, g, b], axis=2)
        top = rgb.max()
        if top > 0:
            rgb = (rgb / top * 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  pseudo-RGB failed: {e}")
        return None


# ── Interactive ROI window ─────────────────────────────────────────────────────

class ROISegmenter:
    """
    Show an image, let the user drag a box, run BiRefNet on the crop,
    display the mask overlay. Returns the final full-size binary mask or None.
    """

    def __init__(self, bgr_img, title="ROI Segmentation"):
        self.orig   = bgr_img.copy()
        self.title  = title
        self.H, self.W = bgr_img.shape[:2]

        # fit large images to screen
        max_side = 900
        self.scale = min(max_side / self.W, max_side / self.H, 1.0)
        self.dW = int(self.W * self.scale)
        self.dH = int(self.H * self.scale)

        self.roi_start  = None   # (x, y) in display coords
        self.roi_end    = None
        self.drawing    = False
        self.mask_full  = None   # H×W uint8
        self.prob_full  = None   # H×W float32

    # ── coord helpers ─────────────────────────────────────────────────────

    def _to_full(self, dx, dy):
        fx = int(np.clip(dx / self.scale, 0, self.W - 1))
        fy = int(np.clip(dy / self.scale, 0, self.H - 1))
        return fx, fy

    def _roi_full(self):
        if self.roi_start is None or self.roi_end is None:
            return None
        x0, y0 = self._to_full(*self.roi_start)
        x1, y1 = self._to_full(*self.roi_end)
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        return x0, y0, x1, y1

    # ── mouse ─────────────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing   = True
            self.roi_start = (x, y)
            self.roi_end   = (x, y)
            self.mask_full = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.roi_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi_end = (x, y)

    # ── BiRefNet ──────────────────────────────────────────────────────────

    def _run_birefnet(self):
        roi = self._roi_full()
        if roi is None:
            print("  ROI too small, draw a larger box")
            return
        x0, y0, x1, y1 = roi
        crop = self.orig[y0:y1, x0:x1]
        print(f"  Running BiRefNet on crop {x1-x0}×{y1-y0}...")
        prob_crop = run_birefnet_on_crop(crop)

        # place back into full image
        prob_full = np.zeros((self.H, self.W), dtype=np.float32)
        prob_full[y0:y1, x0:x1] = prob_crop
        self.prob_full = prob_full
        self.mask_full = (prob_full > 0.5).astype(np.uint8) * 255
        n_px = self.mask_full.sum() // 255
        pct  = n_px / (self.H * self.W) * 100
        print(f"  Mask: {n_px} px ({pct:.1f}%)")

    # ── render ────────────────────────────────────────────────────────────

    def _render(self):
        display = cv2.resize(self.orig, (self.dW, self.dH))

        # green overlay for mask
        if self.mask_full is not None:
            m_small = cv2.resize(self.mask_full, (self.dW, self.dH),
                                 interpolation=cv2.INTER_NEAREST)
            fg = m_small > 127
            display[fg] = (display[fg] * 0.45 +
                           np.array([0, 220, 0]) * 0.55).astype(np.uint8)
            cts, _ = cv2.findContours(m_small, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, cts, -1, (0, 255, 0), 2)

        # ROI rectangle
        if self.roi_start and self.roi_end:
            x0, y0 = self.roi_start
            x1, y1 = self.roi_end
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 200, 255), 2)

        # HUD
        lines = [
            "Drag: draw ROI",
            "Enter/R: run BiRefNet",
            "S: save & next   D: skip   Q: quit",
        ]
        for i, txt in enumerate(lines):
            y = 22 + i * 22
            cv2.putText(display, txt, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(display, txt, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # status
        status = "No mask yet" if self.mask_full is None else \
                 f"Mask ready  ({(self.mask_full > 0).mean()*100:.1f}% coverage)"
        cv2.putText(display, status, (8, self.dH - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(display, status, (8, self.dH - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return display

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self):
        """Returns (mask_uint8 | None, action) where action is 'save'|'skip'|'quit'."""
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.dW, self.dH)
        cv2.setMouseCallback(self.title, self._on_mouse)

        while True:
            cv2.imshow(self.title, self._render())
            key = cv2.waitKey(30) & 0xFF

            if key in (13, ord('r')):          # Enter or R → run
                self._run_birefnet()

            elif key == ord('s'):              # S → save
                cv2.destroyWindow(self.title)
                if self.mask_full is None:
                    print("  No mask to save — run BiRefNet first (Enter/R)")
                    continue
                return self.mask_full, "save"

            elif key == ord('d'):              # D → skip
                cv2.destroyWindow(self.title)
                return None, "skip"

            elif key == ord('q'):              # Q → quit
                cv2.destroyWindow(self.title)
                return None, "quit"


# ── per-artifact helper ───────────────────────────────────────────────────────

def process_one(viewfinder_bgr, name, output_dir):
    """Show ROI window for one image, save mask if accepted. Returns action string."""
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")

    seg   = ROISegmenter(viewfinder_bgr, title=name)
    mask, action = seg.run()

    if action == "save" and mask is not None:
        save_mask(viewfinder_bgr, mask, name, output_dir)

    return action


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewfinder", default=None,
                    help="Single viewfinder PNG path")
    ap.add_argument("--output", default=None,
                    help="Output mask path (single-image mode)")
    ap.add_argument("--batch", action="store_true",
                    help="Batch over all artifact folders that have HSI data")
    ap.add_argument("--output_dir", default="hsi/output",
                    help="Output directory (batch mode, default: hsi/output)")
    ap.add_argument("--dataset", default=str(DATASET_ROOT),
                    help="Dataset root for batch mode (one folder per artifact)")
    ap.add_argument("--artifacts", nargs="+", default=None,
                    help="In batch mode, only process these artifact folders "
                         "(e.g. --artifacts lot1n1 lot2n15)")
    ap.add_argument("--overwrite", action="store_true",
                    help="In batch mode, re-process artifacts that already have a mask "
                         "(default: skip them)")
    ap.add_argument("--hsi_only", action="store_true",
                    help="In batch mode, include every artifact that has HSI data "
                         "(default: require all 4 data types: HSI, photogrammetry, pXRF, Raman)")
    ap.add_argument("--manual", action="store_true",
                    help="Interactive mode: draw the ROI box yourself (for cluttered scenes). "
                         "Default: fully automatic")
    args = ap.parse_args()
    run_one = process_one if args.manual else process_one_auto

    if args.viewfinder:
        # ── single image mode ──────────────────────────────────────────────
        img = cv2.imread(args.viewfinder)
        if img is None:
            sys.exit(f"Cannot read: {args.viewfinder}")
        name = Path(args.viewfinder).stem
        out_dir = str(Path(args.output).parent) if args.output else "."
        action = run_one(img, name, out_dir)
        if action == "save" and args.output:
            # also copy to the exact path requested
            mask_path = Path(out_dir) / f"{name}_mask.png"
            if mask_path.exists() and args.output != str(mask_path):
                import shutil
                shutil.copy(str(mask_path), args.output)

    elif args.batch:
        # ── batch mode ────────────────────────────────────────────────────
        required = {"HSI"} if args.hsi_only else {"HSI", "photogrammetry", "pXRF", "Raman"}
        folders = sorted(
            f for f in Path(args.dataset).iterdir()
            if f.is_dir() and required.issubset(
                {d.name for d in f.iterdir() if d.is_dir()}
            )
        )
        criteria = "HSI data" if args.hsi_only else "all 4 data types"
        if args.artifacts:
            wanted = set(args.artifacts)
            missing = wanted - {f.name for f in folders}
            folders = [f for f in folders if f.name in wanted]
            if missing:
                print(f"WARNING: not found or missing {criteria}: {', '.join(sorted(missing))}")
        print(f"Found {len(folders)} folders with {criteria}")

        out_dir = Path(args.output_dir)

        for i, folder in enumerate(folders):
            name = folder.name

            if not args.overwrite and (out_dir / f"{name}_mask.png").exists():
                print(f"[{i+1}/{len(folders)}] {name} — already has mask, skipping")
                continue

            img, name = find_viewfinder(folder)
            if img is None:
                print(f"[{i+1}/{len(folders)}] {name} — no viewfinder, skipping")
                continue

            print(f"\n[{i+1}/{len(folders)}]", end="")
            action = run_one(img, name, str(out_dir))

            if action == "quit":
                print("Quitting.")
                break
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
