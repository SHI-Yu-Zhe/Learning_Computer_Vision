import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread("test.JPG",0)
plt.imshow(img,cmap='gray',interpolation='bicubic')
plt.xticks([])
plt.yticks([])
plt.show()