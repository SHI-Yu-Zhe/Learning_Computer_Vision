import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG",0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# add sobel filter of derivative x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# add sobel filter of derivative y

image = [img,laplacian,sobelx,sobely]
title = ['original','laplacian','sobel x','sobel y']

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(image[i],cmap='gray')
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])
plt.show()
