"""
ROI-crop BiRefNet segmentation for HSI viewfinder images.

Draw a bounding box around the artifact; the crop is fed to BiRefNet so the
model sees the object up close. The resulting mask is placed back into full-
image coordinates and saved.

Usage — single image:
    python hsi/segment_roi_birefnet.py --viewfinder path/to/viewfinder.png --output path/to/mask.png

Usage — batch over all artifact folders that have HSI:
    python hsi/segment_roi_birefnet.py --batch --output_dir hsi/output

Controls (per image):
    Left-drag   draw / redraw the ROI rectangle
    Enter/R     run BiRefNet on current ROI
    S           save mask and move to next image
    D           discard / skip this image (no file written)
    Q           quit immediately
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForImageSegmentation.from_pretrained(
            _MODEL_ID, trust_remote_code=True)
        _model.to(_device)
        _model.eval()
        print(f"  Loaded on {_device}")
    return _model, _device


def run_birefnet_on_crop(bgr_crop):
    """Run BiRefNet on a BGR crop. Returns float32 prob map at crop size."""
    h, w = bgr_crop.shape[:2]
    rgb   = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb)
    model, device = _load_model()
    tensor = _TRANSFORM(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(tensor)
        if isinstance(preds, (list, tuple)):
            preds = preds[-1]
        preds = torch.sigmoid(preds)
        preds = torch.nn.functional.interpolate(
            preds, size=(h, w), mode="bilinear", align_corners=False)
    return preds.squeeze().cpu().numpy()   # float32 [0,1]


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
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mask_path = out_dir / f"{name}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        # debug overlay
        overlay  = viewfinder_bgr.copy()
        fg       = mask > 127
        overlay[fg] = (overlay[fg] * 0.45 + np.array([0, 220, 0]) * 0.55).astype(np.uint8)
        cts, _   = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cts, -1, (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / f"{name}_mask_overlay.png"), overlay)

        print(f"  Saved: {mask_path}")

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
    ap.add_argument("--overwrite", action="store_true",
                    help="In batch mode, re-process artifacts that already have a mask "
                         "(default: skip them)")
    ap.add_argument("--hsi_only", action="store_true",
                    help="In batch mode, include every artifact that has HSI data "
                         "(default: require all 4 data types: HSI, photogrammetry, pXRF, Raman)")
    args = ap.parse_args()

    if args.viewfinder:
        # ── single image mode ──────────────────────────────────────────────
        img = cv2.imread(args.viewfinder)
        if img is None:
            sys.exit(f"Cannot read: {args.viewfinder}")
        name = Path(args.viewfinder).stem
        out_dir = str(Path(args.output).parent) if args.output else "."
        action = process_one(img, name, out_dir)
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
            f for f in DATASET_ROOT.iterdir()
            if f.is_dir() and required.issubset(
                {d.name for d in f.iterdir() if d.is_dir()}
            )
        )
        criteria = "HSI data" if args.hsi_only else "all 4 data types"
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
            action = process_one(img, name, str(out_dir))

            if action == "quit":
                print("Quitting.")
                break
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
