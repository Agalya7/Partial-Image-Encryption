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
from numpy import zeros
from numpy.linalg import norm
import scipy.misc
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn

def dmd(X, Y, truncate=None):
    U2,Sig2,Vh2 = svd(X, False) # SVD of input matrix
    r = len(Sig2) if truncate is None else truncate # rank truncation
    U = U2[:,:r]
    Sig = diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]
    Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig)) # build A tilde
    mu,W = eig(Atil)
    Phi = dot(dot(dot(Y, V), inv(Sig)), W) # build DMD modes
    return mu, Phi

def check_dmd_result(X, Y, mu, Phi, show_warning=True):
    b = np.allclose(Y, dot(dot(dot(Phi, diag(mu)), pinv(Phi)), X))
    if not b and show_warning:
        warn('dmd result does not satisfy Y=AX')

img_name = "img03"
img = np.float32(cv2.imread(img_name + ".jpg", cv2.IMREAD_COLOR))/255

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
#lab_scaled = np.uint8(255.*(lab - lab.min())/(lab.max() - lab.min()))
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(yuv)
#yuv_scaled = np.uint8(255.*(yuv - yuv.min())/(yuv.max() - yuv.min()))
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
y2, cr, cb = cv2.split(ycrcb)
#ycrcb_scaled = np.uint8(255.*(ycrcb - ycrcb.min())/(ycrcb.max() - ycrcb.min()))
matrix = np.column_stack((b.flatten(), v.flatten(), cr.flatten(), 
                          a.flatten(), u.flatten(), cb.flatten()))
new_col = np.random.permutation(matrix[:, 0])

l_svd = np.array(l)
y_svd = np.array(y)
lu, ls, lvh = np.linalg.svd(l, full_matrices=True)
yu, ys, yvh = np.linalg.svd(y_svd, full_matrices=True)
print(l.shape, lu.shape, ls.shape, lvh.shape)
for rank in range(3, 21):
    svd_decomp = np.matrix(lu[:, :rank]) * np.diag(ls[:rank]) * np.matrix(lvh[:rank, :])
    modified_svd = np.array(svd_decomp.flatten())[0]
    rank_name = str(rank)
    if rank == 3:
        I = np.row_stack((svd_decomp.flatten()))
    else:
        I = np.row_stack((I, svd_decomp.flatten()))
    np.save(img_name + "_decomp.npy", svd_decomp)
    scipy.misc.imsave(img_name + "_decomp.png", svd_decomp)
    os.remove(img_name + "_decomp.npy")
I = I.transpose()
#print (len(I), len(I[0]), len(matrix), len(matrix[0]), I.shape[1]//matrix.shape[1] - 1)
for i in range( I.shape[1]//matrix.shape[1] - 1):
    matrix = np.column_stack((matrix, np.random.permutation(b.flatten()), 
                              np.random.permutation(v.flatten()), 
                              np.random.permutation(cr.flatten()), 
                              np.random.permutation(a.flatten()), 
                              np.random.permutation(u.flatten()), 
                              np.random.permutation(cb.flatten())))
mu, phi = dmd(matrix, I)
print (mu)
