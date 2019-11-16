import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")

kernel = np.ones((9,9),np.uint8)
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)

plt.subplot(1,2,1)
plt.imshow(tophat)
plt.title('tophat')
plt.subplot(1,2,2)
plt.imshow(blackhat)
plt.title('blackhat')
plt.show()