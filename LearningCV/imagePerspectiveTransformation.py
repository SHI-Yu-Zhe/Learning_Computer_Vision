import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("test.JPG",0)
img = img[0:3000,0:3000]

pts1 = np.float32([[500,500],[2500,500],[1000,2500],[2000,2500]])
pts2 = np.float32([[0,3000],[3000,3000],[0,0],[3000,0]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(3000,3000))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("input")
plt.subplot(1,2,2)
plt.imshow(dst)
plt.title("output")
plt.show()
