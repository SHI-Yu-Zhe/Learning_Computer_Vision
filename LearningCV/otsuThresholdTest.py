import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("test.JPG",0)

ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# global thresholding, v==127
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# otsu's thresholding
blur = cv2.GaussianBlur(img,(5,5),0)
# address gaussian filtering to reduce noise
ret3,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

image = [
    img,0,th1,
    img,0,th2,
    img,0,th3
]
for i in range(3):
    plt.subplot(3,3,i*3+1)
    plt.imshow(image[i*3],'gray')
    plt.xticks([]),plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2)
    plt.hist(image[i * 3].ravel(), 256)
    plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3)
    plt.imshow(image[i * 3+2], 'gray')
    plt.xticks([]), plt.yticks([])
plt.show()
