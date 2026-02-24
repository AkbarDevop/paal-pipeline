"""Central configuration for the PAAL pipeline."""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LABEL_DIR  = os.path.join(BASE_DIR, "labels")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

METADATA_CSV        = os.path.join(LABEL_DIR, "metadata.csv")
LABELS_CSV          = os.path.join(LABEL_DIR, "labels.csv")
LABELS_POSTURE4_CSV = os.path.join(LABEL_DIR, "labels_posture4.csv")
PRESENCE_CSV        = os.path.join(LABEL_DIR, "presence_filter.csv")

IMG_SIZE   = 224
MAX_PIG_ID = 19

# 60/20/20 split by pig ID (test pigs are completely held out)
TRAIN_PIG_IDS = list(range(0, 12))
VAL_PIG_IDS   = list(range(12, 16))
TEST_PIG_IDS  = list(range(16, 20))

BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
RANDOM_SEED   = 42

BINARY_CLASSES = {0: "not_standing", 1: "standing"}
POSTURE4_CLASSES = {
    0: "standing",
    1: "sitting",
    2: "lateral_lying",
    3: "sternal_lying",
}
