#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:00:21 2019

@author: agalya
"""

import matplotlib.pyplot as plt
import numpy as np
from Crypto.Cipher import AES
from Crypto.Cipher import Blowfish
from pyDes import *
import random
import string

def StringToList(str_pixel):
    str_pixel = str_pixel[1:-1]
    pixels = []
    data = str_pixel.split("][")
    for i in data:
        pixels.append([int(j) for j in i.split(", ")])
    return pixels

def PlotImage(pixels, shape):
    print ("\nRendering image:")
    img = np.zeros(shape)
    for pixel in pixels:
        x = pixel[0]
        y = pixel[1]
        intensity = pixel[2]
        img[x][y] = intensity
    plt.imshow(img)

def AESAlgoDecrypt(encrypted):
    print ("Calling AES decryption")
    key = b'6#26FRL$ZWD5GS4H'
    a = list(map(chr, range(48, 57)))
    b = list(string.ascii_lowercase)
    c = list(string.ascii_uppercase)
    random.seed(3)
    sample = random.sample(a+b+c, 16)
    iv = ""
    for i in range(16):
        iv += sample[i]
    cfb_decipher = AES.new(key, AES.MODE_CFB, iv)
    str_pixel = cfb_decipher.decrypt(encrypted)
    return str_pixel

def BlowfishAlgoDecrypt(encrypted):
    print ("Calling Blowfish decryption")
    key = b'6#26FRL$ZWD'
    cipher  = Blowfish.new(key, Blowfish.MODE_ECB)
    str_pixel = cipher.decrypt(encrypted)
    return str_pixel

def TripleDESAlgoDecrypt(encrypted):
    print ("Calling TripleDES decryption")
    key = b'6#26FRL$'
    cipher = des("DESCRYPT", CBC, key, pad=None, padmode=PAD_PKCS5)
    str_pixel = cipher.decrypt(encrypted, padmode=PAD_PKCS5)
    return str_pixel

def GotoDecryptAlgo(algo, temp):
    if (algo == 0):
        return AESAlgoDecrypt(temp)
    elif (algo == 1):
        return BlowfishAlgoDecrypt(temp)
    else:
        return TripleDESAlgoDecrypt(temp)

f2 = open("encrypted.txt", "r")
content = f2.read()
f2.close()
content = content.split("*-+")
shape = tuple([int(each) for each in content[0][1:-1].split(", ")])
content = content[1]

splitted = content.split('---')[:-1]
decrypted = []
algo = 0
for each in splitted:
    temp = GotoDecryptAlgo(algo, each)
    decrypted.append(temp)
    algo += 1
    
dec_list = []
for each in decrypted:
    dec_list += (StringToList(each.strip()))
PlotImage(dec_list, shape)
