# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import cv2

def squareMatrix(a, n):
    b = np.zeros((n, n), dtype=np.int32)
    for i in range(len(a)):
        for j in range(len(a[i])):
            b[i][j] = a[i][j]
    return b

#img = mpimg.imread("img01.jpg", )
img = cv2.imread("img01.jpg",cv2.IMREAD_GRAYSCALE)
img = squareMatrix(img, max(img.shape[0], img.shape[1]))
#pad image to center and include the corner pixels in the patches
patches = image.extract_patches_2d(img, (100, 100))
print(patches.shape)
#find average patch
#find distance between each patch and avg patch and normalize by max distance
pca = PCA(n_components=2)
for patch in patches[:1]:
    principalComponents = pca.fit_transform(patch)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    #finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
    print (principalDf)
    
#find M SLIC regions