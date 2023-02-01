from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from skimage import transform
from skimage import exposure
import sys
sys.path.append('../')
sys.path.insert(1, '.')
from CONFIG import config
class SignBoard():
    """
    This is a class for detection and classification models of the traffic signboards.
    """
    def __init__(self,det_path,clas_path,thresh=0):
        """
        This is the constructor for SignBoard class.

        Parameters
        ----------
        det_path : string
        Path to the traffic signboard detector model.
        clas_path : string
            Path to the traffic signboard classification model.
        thresh : float, optional
            threshold value for detection. The default is 0.

        Returns
        -------
        None.

        """
        self.detector = load_model(det_path)
        self.classifier = load_model(clas_path)
        self.predictions = None
        self.label = None
        self.should_draw = True
        self.thresh = thresh
    def bboxout(self,image):
        """
        Parameters
        ----------
        image : 3 channel image (numpy array).
            image input on which to perform detection and classification.

        Returns
        -------
        None.

        """
        image = cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        self.predictions = self.detector.predict(image)[0]

    def crop(self,image):
        """

        Parameters
        ----------
        image : 3 channel image (numpy array).
            image input on which to perform cropping on detected bounding box.

        Returns
        -------
        image : 3 channel image (numpy array).
            cropped image.

        """
        startX = int(self.predictions[0]*image.shape[1]*0.2)
        startY = int(self.predictions[1]*image.shape[0]*0.2)
        endX = int(self.predictions[2]*image.shape[1]*1.2)
        endY = int(self.predictions[3]*image.shape[0]*1.2)
        if endX>image.shape[1]:
            endX = image.shape[1]
        if endY>image.shape[0]:
            endY = image.shape[0]
        image = image[startY:endY,startX:endX,:]
        return image

    def classify(self,image):
        """
        Function to run all inferences on the input image/frame.
        Parameters
        ----------
        image : 3 channel image (numpy array).
            image input on which to perform traffic signboard detection and classification.

        Returns
        -------
        None.

        """
        self.bboxout(image)
        image = self.crop(image)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        preds = self.classifier.predict(image)
        self.label = preds.argmax(axis=1)[0]
        if preds.max(axis=1)[0] < self.thresh:
            self.should_draw = False

    def draw_on_image(self,image,csvpath):
        """
        Function to visualize inference on input image/frame.

        Parameters
        ----------
        image : 3 channel image (numpy array).
            image input on which to draw traffic signboard detection and classification outputs.
        csvpath : string
            path to the csv file containing label names.

        Returns
        -------
        image : 3 channel image (numpy array).
            output image with traffic signboard detection and classification results.

        """
        if self.should_draw:
            w=image.shape[1]
            h=image.shape[0]
            labelNames = open(csvpath).read().strip().split("\n")[1:]
            labelNames = [l.split(",")[1] for l in labelNames]
            tlabel = labelNames[self.label]
            image = cv2.rectangle(image,(int(self.predictions[0]*w),int(self.predictions[1]*h)),(int(self.predictions[2]*w),int(self.predictions[3]*h)),(0,255,0),2)
            image = cv2.putText(image,tlabel,(5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1)
        return image

if __name__=='__main__':
    signb = SignBoard(config.SIGN_DET_MODEL,config.SIGN_CLASS_MODEL)
    video = cv2.VideoCapture('../input/traffic_sign.mp4')
    while video.isOpened():

        ret, frame = video.read()

        if not ret:
            break
        frame_ = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        signb.classify(frame_)
        frame = signb.draw_on_image(frame,config.SIGN_NAMES)

        cv2.imshow('detections',frame)
        if  cv2.waitKey(1) & 0xff == ord('q'):
            break

    print("[INFO] cleaning up...")
    video.release()
    cv2.destroyAllWindows()
