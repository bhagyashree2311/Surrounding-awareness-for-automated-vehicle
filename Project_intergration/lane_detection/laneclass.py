#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import warnings
import sys
sys.path.insert(1, 'C:\\Users\\bhagyashrees\\Documents\\Surrounding-awareness-for-automated-vehicle\\\Project_intergration\\CONFIG')
from CONFIG import config
warnings.filterwarnings("ignore")

class Lanedet:
    """
    class for lane detection task.
    """
    def __init__(self,canny_min,canny_max):
        """
        Constructor for Lanedet class.

        Parameters
        ----------
        canny_min : float
            minimum threshold value for canny edge detection.
        canny_max : float
            maximum threshold value for canny edge detection.

        Returns
        -------
        None.

        """
        self.region_points = None
        self.canny_min = canny_min
        self.canny_max = canny_max
        self.lines=None

    def region_of_interest(self,image):
        """

        Parameters
        ----------
        image : single channel edge detected image

        Returns
        -------
        masked_image : masked image (numpy array)

        """
        # we are going to replace pixels with 0 (black) - the regions we are not interested
        mask = np.zeros_like(image)
        # the region that we are interested in is the lower triangle - 255 white pixels
        cv2.fillPoly(mask, np.array([self.region_points], np.int32), 255)
        # we have to use the mask: we want to keep the regions of the original image where
        # the mask has white colored pixels
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def get_detected_lanes(self,image):
        """
        Function to run lane line detection on input image/frame.
        Parameters
        ----------
        image : 3 channel input image/frame

        Returns
        -------
        image_with_lines : image with detected lanes.

        """

        (height, width) = (image.shape[0], image.shape[1])

        # turn the image into grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # edge detection kernel (Canny's algorithm)
        canny_image = cv2.Canny(gray_image, self.canny_min,self.canny_max)

        # we are interested in the "lower region" of the image (there are the driving lanes)
        self.region_points= [
            (0, height),
            (width / 2, height * 0.75),
            (width, height)
        ]
        # get rid of the un-relevant part of the image
        # just keep the lower triangle region
        cropped_image = self.region_of_interest(canny_image)
        # use the line detection algorithm (radians instead of degrees 1 degree = pi / 180)
        self.lines = cv2.HoughLinesP(cropped_image, rho=config.HOUGH_LINE_RHO, theta=config.HOUGH_LINE_THETA, threshold=config.HOUGH_LINE_threshold,minLineLength=config.HOUGH_LINE_MIN_LINE_LENGTH, maxLineGap=config.HOUGH_LINE_MAX_LINE_GAP)

        # draw the lines on the image
        return self.draw_the_lines(image)

    def draw_the_lines(self,image):
        """
        Parameters
        ----------
        image : 3 channel input image/frame

        Returns
        -------
        image_with_lines : image with detected lanes drawn on input.

        """
        # create a distinct image for the lines [0,255] - all 0 values means black image
        lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # there are (x,y) for the starting and end points of the lines
        if self.lines is not None:
            for line in self.lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        # finally we have to merge the image with the lines
        image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)
        return image_with_lines

if __name__ =="__main__":
    obj = Lanedet(250,350)
    video = cv2.VideoCapture('C:\\Users\\bhagyashrees\\Documents\\Surrounding-awareness-for-automated-vehicle\\\Project_intergration\\input/Lane_detection.mp4')
    while video.isOpened():

        is_grabbed, frame = video.read()

        # because the end of the video
        if not is_grabbed:
            break

        frame = obj.get_detected_lanes(frame)


        cv2.imshow('Lane Detection Video', frame)

        if  cv2.waitKey(1) & 0xff == ord('q'):
            break

    print("[INFO] cleaning up...")
    video.release()
    cv2.destroyAllWindows()
