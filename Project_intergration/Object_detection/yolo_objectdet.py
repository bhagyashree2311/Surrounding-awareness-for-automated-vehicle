import numpy as np
import cv2
import sys
sys.path.insert(1, '../CONFIG')
import config


class Yolo():
    """
    class to load yolov3 pretrained model and run in on input image/frame.
    """
    def __init__(self,configPath, weightsPath, confidencethresh, threshold ,labelpath):
        """
        This is the constructor for Yolo class.

        Parameters
        ----------
        configPath : string
            yolo configuration file path.
        weightsPath : string
            yolo weights file path.
        confidencethresh : float
            threshold value for detections.
        threshold : float
           Non-max supression threshold.
        labelpath : string
            path to coco dataset label names.

        Returns
        -------
        None.

        """
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.boxes = []
        self.idxs = None
        self.classIDs = []
        self.confidences = []
        self.confidencethresh = confidencethresh
        self.threshold = threshold
        self.labels = open(labelpath).read().strip().split("\n")
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

    def forward(self,image):
        """
        Function to run inference on input image/frame.

        Parameters
        ----------
        image : 3 channel input image (numpy array)
            input image/frame to yolo model.

        Returns
        -------
        None.

        """
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        ln = self.net.getLayerNames()
        ln = [ln[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)
        # loop over each of the layer outputs
        for output in layerOutputs:
            #loop over each detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected probability is greater than the minimum probability

                if confidence > self.confidencethresh:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, self.confidences, and class IDs

                    self.boxes.append([x, y, int(width), int(height)])
                    self.confidences.append(float(confidence))
                    self.classIDs.append(classID)
        self.idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.confidencethresh, self.threshold)

    def whatlight(self,image):

        hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Set range for red color and
        # define mask
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180,255,255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        # Set range for green color and
        # define mask
        green_lower = np.array([70,79,137], np.uint8)
        green_upper = np.array([105,255,255], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        # Set range for blue color and
        # define mask
        yellow_lower = np.array([25, 100, 100], np.uint8)
        yellow_upper = np.array([40,255,255], np.uint8)
        yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

        con = {'color_name':'red','area':0,'contour':None,'color':None}
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for _, contour in enumerate(contours):
            if con['area']<cv2.contourArea(contour):
                con['area'] = cv2.contourArea(contour)
                con['color'] = (0,0,255)
                con['contour'] = contour
                con['color_name'] = 'red'

        # Creating contour to track green color
        contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for _, contour in enumerate(contours):
            if con['area']<cv2.contourArea(contour):
                con['area'] = cv2.contourArea(contour)
                con['color'] = (0,255,0)
                con['contour'] = contour
                con['color_name'] = 'green'

        # Creating contour to track yellow color
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for _, contour in enumerate(contours):
            if con['area']<cv2.contourArea(contour):
                con['area'] = cv2.contourArea(contour)
                con['color'] = (0,255,255)
                con['contour'] = contour
                con['color_name'] = 'yellow'
        return con

    def visualize(self, image):
        """
        Function to visualize inferences on imput image.

        Parameters
        ----------
        image : 3 channel input image (numpy array)

        Returns
        -------
        None.

        """

        if len(self.idxs) > 0:
    # loop over the indexes we are keeping
            for i in self.idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])
                if self.classIDs[i] == 9:
                    startx = x
                    starty = y
                    endx =x + w
                    endy =y + h
                    cropimage = image[starty:endy,startx:endx]
                    color = self.whatlight(cropimage)
                    ptx, pty, ptw, pth = cv2.boundingRect(color['contour'])
                    #image = cv2.rectangle(image, (x+ptx, y+pty), (x + ptx + ptw, y + pty + pth), color['color'], 1)
                    cv2.putText(image, color['color_name'], (x+ptx, y+pty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color['color'],1)

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[self.classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[self.classIDs[i]], self.confidences[i])
                cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        self.boxes = []
        self.idxs = None
        self.classIDs = []
        self.confidences = []
        return image


if __name__ == "__main__":
    yolo = Yolo(config.YOLO_CONFIG_PATH,config.YOLO_WEIGHTS_PATH,config.YOLO_CONFIDENCE,config.YOLO_THRESHOLD,config.YOLO_LABELS_PATH)
    video = cv2.VideoCapture('../input/obj_det.mkv')
    while video.isOpened():

        ret, frame = video.read()

        if not ret:
            break

        yolo.forward(frame)
        yolo.visualize(frame)

        cv2.imshow('detections',frame)
        if  cv2.waitKey(1) & 0xff == ord('q'):
            break

    print("[INFO] cleaning up...")
    video.release()
    cv2.destroyAllWindows()
