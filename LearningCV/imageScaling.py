import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("test.JPG")

res1 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
res2 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_AREA)
res3 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)

res = [img,res1,res2,res3]
title = ['original','cubic','area','linear']
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(res[i])
    plt.title(str.upper(title[i]))
    plt.xticks([]),plt.yticks([])
plt.show()
