# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:23:05 2019

@author: 15pt03
"""

#SLIC Superpixels

#import matplotlib.pyplot as plt
import cv2
import math
import numpy as np

def superPixel(img, K):
    m = 10
    N = len(img) * len(img[0])
    S = int(round(math.sqrt(N/K)))
    k_values = np.arange(1, K, S)
    #k_values = [int(round(k)) for k in k_values]
    #print(k_values)
    l, a, b = cv2.split(img)
    C = []
    for k in k_values:
        ky = k % len(img[0])
        kx = int((k - ky) / len(img[0]))
        C.append([l[kx][ky], a[kx][ky], b[kx][ky], kx, ky])
    #print (C)
    D_s = [[0] * len(img[0])] * len(img)
    for k in range(len(C)):
        ky = k % len(img[0])
        kx = int((k - ky) / len(img[0]))
        #specify start as 0 or kx-2*S
        for i in range(kx-2*S, kx+2*S):
            for j in range(ky-2*S, ky+2*S):
                print (kx, ky, i, j)
                d_lab = math.sqrt((C[k][0] - l[i][j])**2 + (C[k][1] - a[i][j])**2 + (C[k][2] - b[i][j])**2)
                d_xy = math.sqrt((C[k][3] - i)**2 + (C[k][4] - j)**2)
                d_s = d_lab + (m/S)*d_xy
                if (d_s < D_s[i][j]):
                    D_s[i][j] = d_s
    #print (D_s[:12][:11])

def SLIC(img):
    l, a, b = cv2.split(img)
    superPixel(img, 500)
    
    #print([i for i, lst in enumerate(img) if [238, 125, 126] in lst][0])
    #print("Index:", img.index([238, 125, 126])
    #for k in k_values:
        #C.append([l[k], a[k], b[k], )
    #for k in k_values:
        #d_tab = math.sqrt((l[k] - l)**2 + ()**2 + ()**2)
    #retrun 0
    
img = cv2.imread("img01.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#plt.imshow(img)
print (img.shape)
SLIC(img)
