"""
Central path configuration for Automata.
All scripts and notebooks should import paths from here instead of hardcoding them.

Usage:
    from paths import DATASET_ROOT, XRF_STANDALONE_ROOT, RAMAN_STANDALONE_ROOT, ...
"""
from pathlib import Path

AUTOMATA_ROOT = Path(__file__).resolve().parent
DATASET_ROOT  = AUTOMATA_ROOT / "dataset"

# ── Unified standalone XRF collections ────────────────────────────────────
XRF_STANDALONE_ROOT      = Path(r"C:\Users\leona\Downloads\xrf_standalone")
XRF_MARTINA_DIR          = XRF_STANDALONE_ROOT / "martina"
XRF_SCIUTO_DIR           = XRF_STANDALONE_ROOT / "sciuto"
XRF_STATUETTE_DIR        = XRF_STANDALONE_ROOT / "statuette_ippopotami"
XRF_STANDARDS_DIR        = XRF_STANDALONE_ROOT / "standards"
XRF_BRUKER_DIR           = XRF_STANDALONE_ROOT / "bruker"

# ── Unified standalone Raman collections ──────────────────────────────────
RAMAN_STANDALONE_ROOT    = Path(r"C:\Users\leona\Downloads\raman_standalone")
RAMAN_LABELLED_BAD_DIR   = RAMAN_STANDALONE_ROOT / "labelled_bad"
RAMAN_LABELLED_GOOD_DIR  = RAMAN_STANDALONE_ROOT / "labelled_good"

# ── Classifier training data ───────────────────────────────────────────────
XRF_BAD_DATA_DIR  = Path(r"C:\Users\leona\Downloads\xrf_bad_data\xrf_bad_data")
RAMAN_BAD_DIR     = RAMAN_LABELLED_BAD_DIR
RAMAN_GOOD_DIR    = RAMAN_LABELLED_GOOD_DIR
