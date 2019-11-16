import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("test.JPG")
b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])
plt.subplot(121)
plt.imshow(img) # expects distorted color
plt.subplot(122)
plt.imshow(img2) # expects true color

cv2.namedWindow("BGR",cv2.WINDOW_NORMAL)
cv2.namedWindow("RGB",cv2.WINDOW_NORMAL)
cv2.imshow("BGR",img) # expects true color
cv2.imshow("RGB",img2) # expects distorted color
cv2.waitKey(0)
cv2.destroyAllWindows()
