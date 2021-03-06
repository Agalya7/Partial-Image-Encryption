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
import scipy.misc

#takes a string as input and returns a list
#'[1, 2][3, 4]' -> [1, 2, 3, 4]
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

def AESAlgoDecrypt(encrypted, key):
    print ("Calling AES decryption")
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

def BlowfishAlgoDecrypt(encrypted, key):
    print ("Calling Blowfish decryption")
    cipher  = Blowfish.new(key, Blowfish.MODE_ECB)
    str_pixel = cipher.decrypt(encrypted)
    return str_pixel

def TripleDESAlgoDecrypt(encrypted, key):
    print ("Calling TripleDES decryption")
    cipher = des("DESCRYPT", CBC, key, pad=None, padmode=PAD_PKCS5)
    str_pixel = cipher.decrypt(encrypted, padmode=PAD_PKCS5)
    return str_pixel

def GotoDecryptAlgo(algo, temp, limit):
    seed1 = 4234
    seed2 = 6736
    seed3 = 7452
    random.seed(seed1)
    key1 = ''.join(random.sample(limit, 16))
    random.seed(seed2)
    key2 = ''.join(random.sample(limit, 16))
    random.seed(seed3)
    key3 = ''.join(random.sample(limit, 8))
    if (algo == 0):
        return AESAlgoDecrypt(temp, key1)
    elif (algo == 1):
        return BlowfishAlgoDecrypt(temp, key2)
    else:
        return TripleDESAlgoDecrypt(temp, key3)

f2 = open("encrypted.txt", "r")
content = f2.read()
f2.close()
content = content.split("*-+")
shape = tuple([int(each) for each in content[0][1:-1].split(", ")])
content = content[1]

splitted = content.split('---')[:-1]
decrypted = []
a = list(map(chr, range(48, 57)))
b = list(string.ascii_lowercase)
c = list(string.ascii_uppercase)
limit = a + b + c
algo = 0
for each in splitted:
    temp = GotoDecryptAlgo(algo, each, limit)
    decrypted.append(temp)
    algo += 1
    
dec_list = []
for each in decrypted:
    dec_list += (StringToList(each.strip()))
PlotImage(dec_list, shape)
