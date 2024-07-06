import os

DATA_FOLDER = "zindi_data"
TRAIN_CSV = os.path.join(DATA_FOLDER, "TrainDataset.csv")
VAL_CSV = os.path.join(DATA_FOLDER, "ValDataset.csv")
TEST_CSV = os.path.join(DATA_FOLDER, "Test.csv")
IMAGE_FOLDER = os.path.join(DATA_FOLDER, "Images/")

NUM_CLASS = 3
MODEL_NAME = "microsoft/conditional-detr-resnet-50"  # hustvl/yolos-small "jozhang97/deta-resnet-50-24-epochs" "hustvl/yolos-base" # "jozhang97/deta-resnet-50" # "facebook/detr-resnet-50"

NMS_THR = 0.75
CLS_THR = 0.25