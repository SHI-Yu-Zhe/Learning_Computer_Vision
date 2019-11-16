import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("test.JPG",0)
img = cv2.medianBlur(img,5)
# address median filtering

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# test of global thresholding
th2 = cv2.adaptiveThreshold(
    img,255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11,4
)
th3 = cv2.adaptiveThreshold(
    img,255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,4
)

title = ['original','global thresholding','adaptive mean','adaptive gaussian']
image = [img,th1,th2,th3]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(image[i],'gray')
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])
plt.show()
