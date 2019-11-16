import cv2
import numpy as np

img = cv2.imread("test.JPG")
img = img[600:1050,600:1050]
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

target = cv2.imread("test.JPG")
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

imghist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])

cv2.normalize(imghist,imghist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],imghist,[0,180,0,256],1)

disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,disc,dst)

ret,thresh = cv2.threshold(dst,50,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)

res = np.vstack((target,thresh,res))
cv2.namedWindow("res",cv2.WINDOW_NORMAL)
cv2.imshow("res",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
