import cv2
import numpy as np

img1 = cv2.imread("test.JPG")
img2 = cv2.imread("drawing.JPG")

rows,cols,channels = img2.shape
roi = img1[0:rows,0:cols]
# to put img2 at top-left corner of img1

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
# set thresh==10, maximum==255
# get mask of img2
mask_inv = cv2.bitwise_not(mask)
# get inverse mask of img2

img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
# black-out the area of img1 in ROI

img2_fg = cv2.bitwise_and(img2,img2,mask=mask)
# take only region of img2 from img2

dst = cv2.add(img1_bg,img2_fg)
img1[0:rows,0:cols] = dst
# put img2 in img1 and modify the main image

cv2.namedWindow("result",cv2.WINDOW_NORMAL)
cv2.imshow("result",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
