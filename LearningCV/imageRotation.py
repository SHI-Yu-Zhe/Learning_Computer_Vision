import cv2
import numpy as np

img = cv2.imread("test.JPG",0)
col,row = img.shape

M = cv2.getRotationMatrix2D((col/2,row/2),90,-1)
dst = cv2.warpAffine(img,M,(col,row))

cv2.namedWindow('res',cv2.WINDOW_NORMAL)
cv2.imshow('res',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

