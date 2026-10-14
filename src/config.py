import os
import torch

# Base Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "Mango Dataset")
PROCESS_DATA_DIR = os.path.join(DATASET_DIR, "process data")
AUGMENTED_DATA_DIR = os.path.join(DATASET_DIR, "Augmented Image")

SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
MASKS_DIR = os.path.join(PROJECT_ROOT, "masks")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
MANUSCRIPT_DIR = os.path.join(PROJECT_ROOT, "manuscript")

for d in [SPLITS_DIR, CHECKPOINTS_DIR, CACHE_DIR, MASKS_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MANUSCRIPT_DIR]:
    os.makedirs(d, exist_ok=True)

# Standardized 8 classes
CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASS_NAMES)}

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image specs
IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Hyperparameters matching Base Paper (Prabhu et al., Table 5)
BATCH_SIZE = 32
TEACHER_LR = 0.001
STUDENT_LR = 0.001
STEP_SIZE = 5
GAMMA = 0.1
WEIGHT_DECAY = 1e-4
SEED = 42

# Knowledge Distillation hyperparameters
KD_ALPHA = 0.5
KD_TEMPERATURE = 4.0
