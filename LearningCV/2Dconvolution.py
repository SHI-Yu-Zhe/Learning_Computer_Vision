import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")

kernel = np.ones((5,5),np.float32)/25
# initial the conbolutional kernal with 5*5 matrix with elements of 1
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(1,2,1)
plt.imshow(img)
plt.title('original')
plt.subplot(1,2,2)
plt.imshow(dst)
plt.title('averaging')
plt.show()
