import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("test.JPG")
col,row,ch = img.shape

pts1 = np.float32([[500,500],[2000,500],[500,2000]])
pts2 = np.float32([[100,1000],[2000,500],[1000,2500]])

M = cv2.getAffineTransform(pts1,pts2)
# create affine transform matrix from pts1 to pts2

dst = cv2.warpAffine(img,M,(col,row))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("input")
plt.subplot(1,2,2)
plt.imshow(dst)
plt.title("output")
plt.show()
