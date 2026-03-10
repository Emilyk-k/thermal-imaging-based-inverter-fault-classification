DATA_PATH = "data/"
TRAIN_DATA = "data/train/"
TEST_DATA = "data/test/"
MODEL_SAVE_PATH = "models/fault_detection_model.h5"
HISTOGRAM_SAVE_PATH = "results/"

IMG_HEIGHT = 640
IMG_WIDTH = 480
CHANNELS = 3
NUM_CLASSES = 7 # 7 -> single key failure

ADVANCED = True
BATCH_SIZE = 2
EPOCHS = 150
LEARNING_RATE = 1e-5
NUM_FROZEN = -30
VALIDATION_SPLIT = 0.2
PATIENCE = 30
LR_MIN = 1e-7
REDUCE_LR_PATIENCE = 5
LR_FACTOR = 0.2

CLASS_NAMES = ['fault0', 'fault1', 'fault2', 'fault3', 'fault4', 'fault5', 'fault6']