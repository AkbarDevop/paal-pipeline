"""
Central configuration for the PAAL pipeline.
Edit paths here if your data lives somewhere else.
"""
import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LABEL_DIR  = os.path.join(BASE_DIR, "labels")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

METADATA_CSV = os.path.join(LABEL_DIR, "metadata.csv")
LABELS_CSV   = os.path.join(LABEL_DIR, "labels.csv")
LABELS_POSTURE4_CSV = os.path.join(LABEL_DIR, "labels_posture4.csv")
PRESENCE_CSV = os.path.join(LABEL_DIR, "presence_filter.csv")

# ── Data format (OAK ToF camera) ──────────────────────
# Each timestamp folder contains per-pig files:
#   pig{ID}_{modality}_{timestamp}.{ext}
# Modalities: depth, depth_vis, ir, ir_vis, rgb
# Extensions: .raw (raw binary), .jpg (visual preview)
#
# For classification we use the .jpg previews.
# For 3D measurement (Task B) we use .raw depth files.

# Image size for model input
IMG_SIZE = 224

# Number of pigs expected per timestamp folder (0-18 = 19 pigs)
# Adjust if your data has fewer
MAX_PIG_ID = 19  # pig0 through pig19 (20 pigs based on scan)

# ── Data split by pig ID ──────────────────────────────
# 60/20/20 split by animal ID (20 pigs total)
# so test set contains pigs the model has NEVER seen.
#
# pig 0-11  = train (12 pigs, 60%)
# pig 12-15 = validation (4 pigs, 20%)
# pig 16-19 = test (4 pigs, 20%, completely held out)
TRAIN_PIG_IDS = list(range(0, 12))    # pig 0-11
VAL_PIG_IDS   = list(range(12, 16))   # pig 12-15
TEST_PIG_IDS  = list(range(16, 20))   # pig 16-19

# ── Training ──────────────────────────────────────────
BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
RANDOM_SEED   = 42

# ── Labels ────────────────────────────────────────────
BINARY_CLASSES = {0: "not_standing", 1: "standing"}
POSTURE4_CLASSES = {
    0: "standing",
    1: "sitting",
    2: "lateral_lying",
    3: "sternal_lying",
}
