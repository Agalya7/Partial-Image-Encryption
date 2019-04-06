# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:59:01 2019

@author: 15pt03
"""

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import scipy.misc
from Crypto.Cipher import AES
from Crypto.Cipher import Blowfish
from pyDes import *
import random
import string

def DMDAlgorithm(l, a, b, u, v, cr, cb):
    matrix = np.column_stack((b.flatten(), v.flatten(), cr.flatten(), 
                              a.flatten(), u.flatten(), cb.flatten()))
    l_svd = np.array(l)
    y_svd = np.array(y)
    lu, ls, lvh = np.linalg.svd(l, full_matrices=True)
    yu, ys, yvh = np.linalg.svd(y_svd, full_matrices=True)
    rank = 20
    svd_decomp = np.matrix(lu[:, :rank]) * np.diag(ls[:rank]) * np.matrix(lvh[:rank, :])
    np.save(img_name + "_decomp.npy", svd_decomp)
    scipy.misc.imsave(img_name + "_decomp1.png", svd_decomp)
    os.remove(img_name + "_decomp.npy")

#takes a list as input and returns a string
#[1, 2, 3, 4] -> '1234'
def ListToString(List):
    str_pixel = ""
    for i in List:
        str_pixel += str(i)
    return str_pixel

def AESAlgoEncrypt(salient_region):
    print ("Calling AES encryption")
    key =  b'6#26FRL$ZWD5GS4H'
    a = list(map(chr, range(48, 57)))
    b = list(string.ascii_lowercase)
    c = list(string.ascii_uppercase)
    random.seed(3)
    sample = random.sample(a+b+c, 16)
    iv = ""
    for i in range(16):
        iv += sample[i]
    cfb_cipher = AES.new(key, AES.MODE_CFB, iv)
    str_pixel = ListToString(salient_region)
    encrypted = cfb_cipher.encrypt(str_pixel)
    return encrypted
    
#def RSAAlgoEncrypt(salient_region):
#    modulus_length = 1024
#    key = RSA.generate(modulus_length)
#    public_key = key.publickey()
#    str_pixel = ListToString(salient_region)
#    print (len(str_pixel.encode('utf-8')))
#    encryptor = PKCS1_OAEP.new(key)
#    #upto length 86 can encrypt
#    encrypted = encryptor.encrypt(str_pixel)
#    encryptor = PKCS1_OAEP.new(public_key)
#    str_pixel = encryptor.decrypt(encrypted)
#    pixels = StringToList(str_pixel)
#    PlotImage(pixels, shape)
#    return pixels
    
def BlowfishAlgoEncrypt(salient_region):
    print ("Calling Blowfish encryption")
    str_pixel = ListToString(salient_region)
    byteNum = len(str_pixel)
    packingLength = 8 - byteNum % 8
    appendage = ' ' * packingLength
    key = b'6#26FRL$ZWD'
    cipher  = Blowfish.new(key, Blowfish.MODE_ECB)
    packedString = str_pixel + appendage
    encrypted = cipher.encrypt(packedString)
    return encrypted
    
def TripleDESAlgoEncrypt(salient_region):
    print ("Calling TripleDES encryption")
    str_pixel = ListToString(salient_region)
    key = b'6#26FRL$'
    cipher = des("DESCRYPT", CBC, key, pad=None, padmode=PAD_PKCS5)
    encrypted = cipher.encrypt(str_pixel)
    return encrypted

def GotoEncryptAlgo(algo, temp):
    if (algo == 0):
        return AESAlgoEncrypt(temp)
    elif (algo == 1):
        return BlowfishAlgoEncrypt(temp)
    else:
        return TripleDESAlgoEncrypt(temp)

#read image and split based on color space
img_name = "img02"
img = cv2.imread(img_name + ".jpg", cv2.IMREAD_COLOR)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(yuv)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
y2, cr, cb = cv2.split(ycrcb)
DMDAlgorithm(l, a, b, u, v, cr, cb)

img = cv2.imread(img_name + ".jpg", cv2.IMREAD_GRAYSCALE)
print "Size of image:", len(img), len(img[0])
salient_region = []
pixel = []
for i in range(len(img)):
    for j in range(len(img[0])):
        if img[i][j] < 127:
            img[i][j] = 0
        else:
            salient_region.append([i, j, img[i][j]])
scipy.misc.imsave(img_name + "_decomp2.png", img)
print "Total pixels to be ecncrypted:", len(salient_region)
length = random.sample(range(1, 4), 1)[0]
print "Number of segments:", length

algorithm = np.zeros((len(salient_region)))
for i in range(len(algorithm)):
    algorithm[i] = random.randint(0, 9999) % length

enc = ""
for algo in range(length):
    temp = []
    for j in range(len(algorithm)):
        if (algorithm[j] == algo):
            temp.append(salient_region[j])
    temp = GotoEncryptAlgo(algo, temp)
    enc += temp + "---"

enc = str(img.shape) + "*-+" + enc
f1 = open("encrypted.txt", "w")
f1.write(enc)
f1.close()
