import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")
kernel = np.ones((5,5),np.uint8)
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)

plt.subplot(1,2,1)
plt.imshow(img)
plt.title('original')
plt.subplot(1,2,2)
plt.imshow(gradient)
plt.title('gradient')
plt.show()
