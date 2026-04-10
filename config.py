"""Central configuration for the PAAL pipeline."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")


def _default_data_dir():
    """Prefer data/ when present, otherwise fall back to images/."""
    data_dir = os.path.join(BASE_DIR, "data")
    if os.path.isdir(data_dir):
        return data_dir
    if os.path.isdir(IMAGE_DIR):
        return IMAGE_DIR
    return data_dir


DATA_DIR = _default_data_dir()
LABEL_DIR = os.path.join(BASE_DIR, "labels")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

METADATA_CSV        = os.path.join(LABEL_DIR, "metadata.csv")
LABELS_CSV          = os.path.join(LABEL_DIR, "labels.csv")
LABELS_POSTURE3_CSV = os.path.join(LABEL_DIR, "labels_posture3.csv")
PRESENCE_CSV        = os.path.join(LABEL_DIR, "presence_filter.csv")
VULVA_LABELS_CSV    = os.path.join(LABEL_DIR, "vulva_labels.csv")
VULVA_MASK_DIR      = os.path.join(LABEL_DIR, "vulva_masks")
VULVA_DEPTH_CSV     = os.path.join(LABEL_DIR, "vulva_depth_labels.csv")
VULVA_DEPTH_MASK_DIR = os.path.join(LABEL_DIR, "vulva_masks_depth")
VULVA_DEPTHMAP_DIR  = os.path.join(BASE_DIR, "depthmap_vulva")
VULVA_MANUAL_LABELS_CSV = os.path.join(LABEL_DIR, "vulva_manual_labels.csv")
VULVA_MANUAL_MASK_DIR = os.path.join(LABEL_DIR, "vulva_masks_manual")
VULVA_MANUAL_MEASUREMENTS_CSV = os.path.join(LABEL_DIR, "vulva_measurements_manual.csv")
VULVA_LENGTH_WIDTH_CSV = os.path.join(LABEL_DIR, "vulva_length_width_labels.csv")

IMG_SIZE   = 224

# Crop box to remove neighbor pigs and ceiling (in ToF 640x480 coords)
# For RGB 1280x800, scale by (w/640, h/480)
CROP_TOF = (120, 30, 500, 480)  # (x_left, y_top, x_right, y_bottom)

# 60/20/20 split by pig ID (test pigs are completely held out)
TRAIN_PIG_IDS = list(range(0, 12))
VAL_PIG_IDS   = list(range(12, 16))
TEST_PIG_IDS  = list(range(16, 20))

BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4

BINARY_CLASSES = {0: "not_standing", 1: "standing"}
POSTURE3_CLASSES = {0: "standing", 1: "sitting", 2: "lying"}


def resolve_path(csv_path):
    """Resolve an absolute path from CSV to work on any machine.

    Label CSVs store absolute paths (e.g. /Users/akbar/.../data/...).
    If the path doesn't exist (different machine), extract the relative
    part after '/data/' and resolve against the local DATA_DIR.
    """
    if not csv_path:
        return csv_path

    if os.path.exists(csv_path):
        return csv_path

    if not os.path.isabs(csv_path):
        local_rel = os.path.join(BASE_DIR, csv_path)
        if os.path.exists(local_rel):
            return local_rel

    parts = csv_path.replace("\\", "/").split("/data/")
    if len(parts) == 2:
        rel = parts[1]
        candidates = []
        for root in (DATA_DIR, IMAGE_DIR, os.path.join(BASE_DIR, "data")):
            if root and root not in candidates:
                candidates.append(root)
        for root in candidates:
            candidate = os.path.join(root, rel)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(DATA_DIR, rel)

    return csv_path
