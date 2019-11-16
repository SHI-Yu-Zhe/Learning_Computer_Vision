import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("test.JPG",0)
col,row = img.shape
# get initial width and height of image

M = np.float32([[1,0,600],[0,1,400]])
# matrix M as an transformer
dst = cv2.warpAffine(img,M,(col,row))
# create an affine from original image to new image

cv2.namedWindow('res',cv2.WINDOW_NORMAL)
cv2.imshow('res',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
