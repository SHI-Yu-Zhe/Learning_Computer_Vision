import cv2
import numpy as np

# load a color image in gray scale
img = cv2.imread('test.JPG',-1)

# show the image
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(10000)
cv2.destroyAllWindows()