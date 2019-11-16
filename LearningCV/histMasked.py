import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("test.JPG",0)

mask = np.zeros(img.shape[:2],np.uint8)
mask[1000:2500,1000:3000] = 255
mask_img = cv2.bitwise_and(img,img,mask=mask)

hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
h = [img,mask,mask_img]
title = ['image','mask','masked_img']
for i in range(len(h)):
    plt.subplot(2,2,i+1)
    plt.imshow(h[i],'gray')
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])
plt.subplot(2,2,4)
plt.plot(hist_full),plt.plot(hist_mask)
plt.title('histogram')
plt.show()
