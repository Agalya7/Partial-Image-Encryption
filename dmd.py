# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:59:01 2019

@author: 15pt03
"""

import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
from numpy.linalg import norm
import scipy.misc

img = np.float32(cv2.imread("img03.jpg", cv2.IMREAD_COLOR))/255

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

#print(len(img), len(img[0]))
#print (matrix1)

l_svd = np.array(l)
y_svd = np.array(y)
lu, ls, lvh = np.linalg.svd(l, full_matrices=True)
yu, ys, yvh = np.linalg.svd(y_svd, full_matrices=True)
print(l.shape, lu.shape, ls.shape, lvh.shape)
rank = 20
svd_decomp = np.matrix(lu[:, :rank]) * np.diag(ls[:rank]) * np.matrix(lvh[:rank, :])
print(svd_decomp)
np.save("decomposed.npy", svd_decomp)
scipy.misc.imsave("decomposed.png", svd_decomp)
#print(np.allclose(l_svd, np.dot(lu[:, :l.shape[1]] * ls, lvh)))
#smat = np.zeros((lu.shape, lvh.shape), dtype=complex)

#print (np.diag(ls).shape, lvh.shape)  #<- here is the problem - size do not match
#p1 = np.dot(np.diag(ls), lvh)
#assert np.allclose(l_svd, np.dot(lu, p1))
#
#ls[2:] = 0
#new_a = np.dot(lu, np.dot(np.diag(ls), lvh))
#print(new_a)