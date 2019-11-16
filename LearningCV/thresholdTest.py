import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("test.JPG",0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_OTSU)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh6 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

image = [thresh1,thresh2,thresh3,thresh4,thresh5,thresh6]
title = ['BINARY','BINARY_INV','THRESH_TRUNC','THRESH_OTSU','THRESH_TOZERO','THRESH_TOZERO_INV']

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(image[i],'gray')
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])
    # add nothing to x,y axes
plt.show()
