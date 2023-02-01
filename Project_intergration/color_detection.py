#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

image = cv2.imread('./input/tf4.jpg')


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
contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
for pic, contour in enumerate(contours):
    if con['area']<cv2.contourArea(contour):
        con['area'] = cv2.contourArea(contour)
        con['color'] = (0,0,255)
        con['contour'] = contour
        con['color_name'] = 'red'
        
# Creating contour to track green color
contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
for pic, contour in enumerate(contours):
    if con['area']<cv2.contourArea(contour):
        con['area'] = cv2.contourArea(contour)
        con['color'] = (0,255,0)
        con['contour'] = contour
        con['color_name'] = 'green'
        
# Creating contour to track yellow color
contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    if con['area']<cv2.contourArea(contour):
        con['area'] = cv2.contourArea(contour)
        con['color'] = (0,255,255)
        con['contour'] = contour
        con['color_name'] = 'yellow'
   
x, y, w, h = cv2.boundingRect(con['contour'])

image = cv2.rectangle(image, (x, y), (x + w, y + h), con['color'], 2)
  
cv2.putText(image, con['color_name'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
        
cv2.imshow("Color Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()