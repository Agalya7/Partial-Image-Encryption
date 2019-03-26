# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:59:01 2019

@author: 15pt03
"""

import matplotlib.pyplot as plt
import cv2
import math
import os
import numpy as np
from numpy.linalg import norm
import scipy.misc

img_name = "img03"
img = np.float32(cv2.imread(img_name + ".jpg", cv2.IMREAD_COLOR))/255

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
lab_scaled = np.uint8(255.*(lab - lab.min())/(lab.max() - lab.min()))
#l_scaled = np.uint8(255.*(l - l.min())/(l.max() - l.min()))
#a_scaled = np.uint8(255.*(a - a.min())/(a.max() - a.min()))
#b_scaled = np.uint8(255.*(b - b.min())/(b.max() - b.min()))
#cv2.imshow('LAB', l_scaled)
#cv2.waitKey(0)
#cv2.imshow('LAB', a_scaled)
#cv2.waitKey(0)
#cv2.imshow('LAB', b_scaled)
#cv2.waitKey(0)

yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(yuv)
yuv_scaled = np.uint8(255.*(yuv - yuv.min())/(yuv.max() - yuv.min()))
#cv2.imshow('YUV', yuv_scaled)
#cv2.waitKey(0)
#y_scaled = np.uint8(255.*(y - y.min())/(y.max() - y.min()))
#u_scaled = np.uint8(255.*(u - u.min())/(u.max() - u.min()))
#v_scaled = np.uint8(255.*(v - v.min())/(v.max() - v.min()))
#cv2.imshow('LAB', y_scaled)
#cv2.waitKey(0)
#cv2.imshow('LAB', u_scaled)
#cv2.waitKey(0)
#cv2.imshow('LAB', v_scaled)
#cv2.waitKey(0)

ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
y2, cr, cb = cv2.split(ycrcb)
ycrcb_scaled = np.uint8(255.*(ycrcb - ycrcb.min())/(ycrcb.max() - ycrcb.min()))
#cv2.imshow('YCrCb', ycrcb_scaled)
#cv2.waitKey(0)
#y2_scaled = np.uint8(255.*(y2 - y2.min())/(y2.max() - y2.min()))
#cr_scaled = np.uint8(255.*(cr - cr.min())/(cr.max() - cr.min()))
#cb_scaled = np.uint8(255.*(cb - cb.min())/(cb.max() - cb.min()))
#cv2.imshow('LAB', y2_scaled)
#cv2.waitKey(0)
#cv2.imshow('LAB', cr_scaled)
#cv2.waitKey(0)
#cv2.imshow('LAB', cb_scaled)
#cv2.waitKey(0)

matrix1 = np.column_stack((b.flatten(), v.flatten(), cr.flatten()))
matrix2 = np.column_stack((a.flatten(), u.flatten(), cb.flatten()))
#print (matrix1[:, 0])
new_col = np.random.permutation(matrix1[:, 0])
#print(new_col[:100])
#print(len(img), len(img[0]))
#print (matrix1)

l_svd = np.array(l)
y_svd = np.array(y)
lu, ls, lvh = np.linalg.svd(l, full_matrices=True)
yu, ys, yvh = np.linalg.svd(y_svd, full_matrices=True)
print(l.shape, lu.shape, ls.shape, lvh.shape)
I = [[0 for i in range(20-3+1)] for j in range(l.shape[0] * l.shape[1])]
col = 0
for rank in range(3, 4):
    svd_decomp = np.matrix(lu[:, :rank]) * np.diag(ls[:rank]) * np.matrix(lvh[:rank, :])
    mofified_svd = np.array(svd_decomp.flatten())[0]
    print (len(I), len(I[0]), len(mofified_svd))
    #I = np.hstack((I, mofified_svd))
    #np.append(I, svd_decomp.flatten(), axis=1)
    #print(svd_decomp)
    rank_name = str(rank)
    file_name = img_name + "_" + rank_name
    np.save(file_name + "_decomp.npy", svd_decomp)
    scipy.misc.imsave(file_name + "_decomp.png", svd_decomp)
    os.remove(file_name + "_decomp.npy")
    
    col += 1
#print ([row[0] for row in I])
    #lu, ls, lvh = np.linalg.svd(svd_decomp, full_matrices=True)
#print(np.allclose(l_svd, np.dot(lu[:, :l.shape[1]] * ls, lvh)))
#smat = np.zeros((lu.shape, lvh.shape), dtype=complex)

#print (np.diag(ls).shape, lvh.shape)  #<- here is the problem - size do not match
#p1 = np.dot(np.diag(ls), lvh)
#assert np.allclose(l_svd, np.dot(lu, p1))
#
#ls[2:] = 0
#new_a = np.dot(lu, np.dot(np.diag(ls), lvh))
#print(new_a)
