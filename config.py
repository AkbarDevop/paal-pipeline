"""Central configuration for the PAAL pipeline."""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LABEL_DIR  = os.path.join(BASE_DIR, "labels")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

METADATA_CSV        = os.path.join(LABEL_DIR, "metadata.csv")
LABELS_CSV          = os.path.join(LABEL_DIR, "labels.csv")
LABELS_POSTURE3_CSV = os.path.join(LABEL_DIR, "labels_posture3.csv")
PRESENCE_CSV        = os.path.join(LABEL_DIR, "presence_filter.csv")
VULVA_LABELS_CSV    = os.path.join(LABEL_DIR, "vulva_labels.csv")
VULVA_MASK_DIR      = os.path.join(LABEL_DIR, "vulva_masks")

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
    if not csv_path or os.path.exists(csv_path):
        return csv_path
    parts = csv_path.replace("\\", "/").split("/data/")
    if len(parts) == 2:
        return os.path.join(DATA_DIR, parts[1])
    return csv_path
