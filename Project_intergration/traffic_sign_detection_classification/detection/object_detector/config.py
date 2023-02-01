# import the necessary packages
import os
# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH,"Train","images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model, label binarizer, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 0.0001
NUM_EPOCHS = 70
BATCH_SIZE = 16
