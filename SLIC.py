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
    val = m/S
    l, a, b = cv2.split(img)
    C = []
    prev = 0
    threshold = 0.5
    for k in range(K):
        t = k * S
        ky = t % len(img[0])
        kx = int((t - ky) / len(img[0]))
        C.append([l[kx][ky], a[kx][ky], b[kx][ky], kx, ky])
    while(1):
        D_s = [[float('inf')] * len(img[0])] * len(img)
        C_c = [[0] * len(img[0])] * len(img)
        C_sum = [[0] * len(C[0])] * len(C)
        k_sum = [0] * K
        new_img = [[0] * len(img[0])] * len(img)
        for k in range(K):
            t = k * S
            ky = t % len(img[0])
            kx = int((t - ky) / len(img[0]))
            c0 = C[k][0]
            c1 = C[k][1]
            c2 = C[k][2]
            c3 = C[k][3]
            c4 = C[k][4]
            #specify start as 0 or kx-2*S
            istart = kx-2*S if (kx-2*S) >= 0 else 0
            iend = kx+2*S if (kx+2*S) < len(img) else len(img) - 1
            jstart = ky-2*S if (ky-2*S) >= 0 else 0
            jend = ky+2*S if (ky+2*S) < len(img[0]) else len(img[0]) - 1
            #instead of calculating for every pixel, we choose this particular range
            #print (istart, iend, jstart, jend)
            for i in range(istart, iend):
                #print (istart, iend, jstart, jend)
                for j in range(jstart, jend):
                    #print (i, j)
                    d_lab = math.sqrt((c0 - l[i][j])**2 + (c1 - a[i][j])**2 + (c2 - b[i][j])**2)
                    d_xy = math.sqrt((c3 - i)**2 + (c4 - j)**2)
                    d_s = d_lab + val*d_xy
                    #print (d_lab, d_xy, d_s)
                    if (d_s < D_s[i][j]):
                        D_s[i][j] = d_s
                        C_c[i][j] = k
                        new_img[i][j] = C[k][:3]
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(D_s[i][j] == float('inf')):
                    D_s[i][j] = D_s[i-1][j-1]
        #print (D_s[:100][:100])
        current = np.sum(D_s)
        print(current, np.sum(D_s))
        
        #calculating new clusters
        for i in range(len(img)):
            for j in range(len(img[0])):
                k = C_c[i][j]
                C_sum[k][0] += l[i][j]
                C_sum[k][1] += a[i][j]
                C_sum[k][2] += b[i][j]
                C_sum[k][3] += i
                C_sum[k][4] += j
                k_sum[k] += 1
        for k in range(K):
            if (k_sum[k] > 0):
                C[k][:] = [x / k_sum[k] for x in C_sum[k]]
            else:
                C[k][:] = [0*x for x in C_sum[k]]
        if not (prev == 0):
            if (((current - prev) / prev) > threshold):
                break
        prev = current
        print((current - prev) / prev)
    print (new_img[:10][:10])
    #cv2.imshow('Complex', new_img)

def SLIC(img):
    l, a, b = cv2.split(img)
    superPixel(img, 50)
    
img = cv2.imread("img01.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#plt.imshow(img)
print (img.shape)
SLIC(img)
