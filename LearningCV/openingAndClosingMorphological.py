import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

plt.subplot(1,2,1)
plt.imshow(opening)
plt.title('opening')
plt.subplot(1,2,2)
plt.imshow(closing)
plt.title('closing')
plt.show()