import os
from numpy import pi
SIGN_DET_MODEL = "C:\\Users\\bhagyashrees\\Documents\\Surrounding-awareness-for-automated-vehicle\\\Project_intergration\\traffic_sign_detection_classification\\detection\\output\\detector.h5"
SIGN_CLASS_MODEL = "C:\\Users\\bhagyashrees\\Documents\\Surrounding-awareness-for-automated-vehicle\\\Project_intergration\\traffic_sign_detection_classification\\classifier\\output\\trafficnet.model"
SIGN_NAMES = "C:\\Users\\bhagyashrees\\Documents\\Surrounding-awareness-for-automated-vehicle\\\Project_intergration\\traffic_sign_detection_classification\\classifier\\signnames.csv"
OUTPUT = "C:\\Users\\bhagyashrees\\Documents\\Surrounding-awareness-for-automated-vehicle\\\Project_intergration\\output"


YOLO_DIRECTORY = "C:\\Users\\bhagyashrees\\Documents\\Surrounding-awareness-for-automated-vehicle\\\Project_intergration\\Object_detection\\yolo-coco"
YOLO_WEIGHTS_PATH = os.path.sep.join([YOLO_DIRECTORY, "yolov3.weights"])
YOLO_CONFIG_PATH = os.path.sep.join([YOLO_DIRECTORY, "yolov3.cfg"])
YOLO_LABELS_PATH = os.path.sep.join([YOLO_DIRECTORY, "coco.names"])

YOLO_THRESHOLD = 0.2
YOLO_CONFIDENCE = 0.8

CANNY_MIN_THRESH = 100
CANNY_MAX_THRESH = 150

HOUGH_LINE_RHO = 1
HOUGH_LINE_THETA = pi/180
HOUGH_LINE_threshold = 20
HOUGH_LINE_MIN_LINE_LENGTH = 40
HOUGH_LINE_MAX_LINE_GAP = 40
