import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")
g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
g = np.float32(g)

dst = cv2.cornerHarris(g,2,3,0.04)

dst = cv2.dilate(dst,None)

img[dst > 0.01*dst.max()] = [0,0,255]
plt.imshow(img)
plt.show()
