import os
import cv2
from CONFIG import config
from traffic_sign_detection_classification.traffic_signboard_detection import SignBoard
from Object_detection.yolo_objectdet import Yolo
from lane_detection.laneclass import Lanedet
import sys
sys.path.insert(1, './CONFIG')
ROOT_DIR = os.path.abspath(os.curdir)

signb = SignBoard(config.SIGN_DET_MODEL,config.SIGN_CLASS_MODEL,0.8)
objdet = Yolo(config.YOLO_CONFIG_PATH,config.YOLO_WEIGHTS_PATH,config.YOLO_CONFIDENCE,config.YOLO_THRESHOLD,config.YOLO_LABELS_PATH)
lane = Lanedet(150,200)

video = cv2.VideoCapture('./input/obj_det.mkv')

while video.isOpened():

    ret, frame = video.read()

    if not ret:
        break
    frame_ = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    signb.classify(frame_)
    objdet.forward(frame)
    frame = lane.get_detected_lanes(frame)
    frame = signb.draw_on_image(frame,config.SIGN_NAMES)
    objdet.visualize(frame)

    cv2.imshow('detections',frame)
    if  cv2.waitKey(1) & 0xff == ord('q'):
        break
print("[INFO] cleaning up...")
video.release()
cv2.destroyAllWindows()
