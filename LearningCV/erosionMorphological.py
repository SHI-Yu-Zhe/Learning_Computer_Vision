import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations=1)

plt.subplot(1,2,1)
plt.imshow(img)
plt.title('original')
plt.subplot(1,2,2)
plt.imshow(erosion)
plt.title('erosion')
plt.show()