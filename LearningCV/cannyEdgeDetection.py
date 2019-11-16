import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG",0)
for i in range(9):
    if i==0:
        plt.subplot(331)
        plt.imshow(img,cmap='gray')
        plt.title('original')
        plt.xticks([]),plt.yticks([])
    else:
        minVal = 60+i*10
        maxVal = 120+i*10
        edges = cv2.Canny(img,minVal,maxVal)
        plt.subplot(3,3,i+1)
        plt.imshow(edges,cmap='gray')
        plt.title('min='+str(minVal)+','+'max='+str(maxVal))
        plt.xticks([]),plt.yticks([])

plt.show()