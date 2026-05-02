import torch

# Image
IMAGE_SIZE     = (50, 50)
RESIZE_X       = 50
RESIZE_Y       = 50
INPUT_CHANNELS = 3
MEAN           = [0.5, 0.5, 0.5]
STD            = [0.5, 0.5, 0.5]

# Dataset
FULL_DATASET_PATH = "/kaggle/input/datasets/paultimothymooney/breast-histopathology-images"
SAMPLES_PER_CLASS = 6000

# Split
TRAIN_VAL_TEST_SPLIT = (0.70, 0.15, 0.15)
RANDOM_SEED          = 42

# Training
BATCH_SIZE    = 64
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 2

# Scheduler
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR   = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 5

# Classes
CLASSES     = {0: "IDC Negative", 1: "IDC Positive"}
NUM_CLASSES = 2

# Paths
CHECKPOINT_DIR = "checkpoints/"
WEIGHTS_PATH   = "checkpoints/final_weights.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# interface.py compatibility aliases
batchsize = BATCH_SIZE
epochs    = NUM_EPOCHS

print(f"Device           : {DEVICE}")
print(f"Batch size       : {BATCH_SIZE}")
print(f"Epochs           : {NUM_EPOCHS}")
print(f"Samples/class    : {SAMPLES_PER_CLASS}")
