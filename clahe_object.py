# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:43:04 2020

@author: SYunMoon
"""

import numpy as np
import cv2
from skimage import io

image = io.imread('../image/lacnormal.png')

kernel2 = np.ones((2,2),np.uint8)
kernel3 = np.ones((3,3),np.uint8)
kernel4 = np.ones((4,4),np.uint8)
kernel5 = np.ones((5,5),np.uint8)

tophat2 = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel2)
tophat3 = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel3)
tophat4 = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel4)
tophat5 = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel5)

blackhat2 = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel2)
blackhat3 = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel3)
blackhat4 = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel4)
blackhat5 = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel5)

sum_tophat = tophat2 + tophat3 + tophat4 + tophat5
sum_blackhat = blackhat2 + blackhat3 + blackhat4 + blackhat5

final_image = image+(0.5 * sum_tophat)-(0.5 * sum_blackhat)

cv2.imshow('processed', final_image/255)