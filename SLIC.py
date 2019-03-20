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
from numpy.linalg import norm

def superPixel(img, K):
    m = 10
    N = len(img) * len(img[0])
    S = int(round(math.sqrt(N/K)))
    val = m/S
    l, a, b = cv2.split(img)
    C = []
    prev = 0
    threshold = 0.5
    print (len(img), len(img[0]))
    index, kx, ky = 0, 0, 0
    for k in range(K):
        t = k * S
        #ky = t % len(img[0])
        ky += S
        #kx = int((t - ky) / len(img[0]))
        kx = index*S
        if (abs(ky - len(img[0])) < S):
            ky = 0
            index += 1
        #print (kx, ky)
        C.append([l[kx][ky], a[kx][ky], b[kx][ky], kx, ky])
    #print(C)
    while(1):
        D_s = [[float('inf') for i in range(len(img[0]))] for j in range(len(img))]
        C_c = [[0 for i in range(len(img[0]))] for j in range(len(img))]
        C_sum = [[0 for i in range(len(C[0]))] for j in range(len(C))]
        k_sum = [0 for i in range(K)]
        new_img = [[0 for i in range(len(img[0]))] for j in range(len(img))]
        index, kx, ky = 0, 0, 0
        for k in range(K):
            t = k * S
            #ky = t % len(img[0])
            ky += S
            #kx = int((t - ky) / len(img[0]))
            kx += index*S
            if (abs(ky - len(img[0])) < S):
                ky = 0
                index += 1
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
            #print (istart, iend, jstart, jend)
            #instead of calculating for every pixel, we choose this particular range
            for i in range(istart, iend):
                for j in range(jstart, jend):
                    d_lab = math.sqrt((c0 - l[i][j])**2 + (c1 - a[i][j])**2 + (c2 - b[i][j])**2)
                    d_xy = math.sqrt((c3 - i)**2 + (c4 - j)**2)
                    d_s = d_lab + val*d_xy
                    if (d_s < D_s[i][j]):
                        D_s[i][j] = d_s
                        C_c[i][j] = k
                        img[i][j] = C[k][:3]
        for i in range(len(img)):
            for j in range(len(img[0])):
                k = (j - 1) if j>(len(img)//2) else j+1
                while(np.isinf(D_s[i][j])):
                    D_s[i][j] = D_s[i][k]
                    k -= 1
        #print (D_s[:100][:100])
        #too many inf values in D_s
        current = np.sum(D_s)
        print(current, np.isinf(D_s))       #still getting inf values
        #print (C_c[:100][:100])
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
        #print ("C_SUM:", C_sum)
        for k in range(K):
            if (k_sum[k] > 0):
                C[k][:] = [x / k_sum[k] for x in C_sum[k]]
            else:
                C[k][:] = [0*x for x in C_sum[k]]
        #print ("->", C, np.sum(k_sum))
        if not (prev == 0):
            if (((current - prev) / prev) > threshold):
                break
        prev = current
        #print((current - prev) / prev)
    #print (img[:3][:3])
    #cv2.imshow('Complex', img)

def SLIC(img):
    l, a, b = cv2.split(img)
    superPixel(img, 500)
    
img = cv2.imread("img01.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#plt.imshow(img)
print (img.shape)
SLIC(img)
