import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")

averageBlur = cv2.blur(img,(5,5))
gaussianBlur = cv2.GaussianBlur(img,(5,5),0)
medianBlur = cv2.medianBlur(img,5)
bilateralFilter = cv2.bilateralFilter(img,9,75,75)

res = [averageBlur,gaussianBlur,medianBlur,bilateralFilter]
title = ['averageBlur','gaussianBlur','medianBlur','bilateralFilter']

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(res[i])
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])
plt.show()
