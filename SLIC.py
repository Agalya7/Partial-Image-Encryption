# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:23:05 2019

@author: 15pt03
"""

#SLIC Superpixels

import matplotlib.pyplot as plt
import cv2
import math
import numpy as np

def SLIC(img, K):
    N = img.shape[0] * img.shape[1]
    S = math.sqrt(N/K)
    print(S)
    print(np.arange(1, K, S))
    #retrun 0
    
img = cv2.imread("img01.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#plt.imshow(img)
print (img.shape)
l_channel, a_channel, b_channel = cv2.split(img)
#plt.imshow(l_channel)
#plt.imshow(a_channel)
#plt.imshow(b_channel)
SLIC(img, max(img.shape[0], img.shape[1]))