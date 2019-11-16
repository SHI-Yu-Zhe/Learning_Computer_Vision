import numpy as np
import cv2

img = cv2.imread("test.JPG")
sec = img[250:1000,250:1000]
# select a pixel matrix as a new object
img[1000:1750,1000:1750] = sec
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()