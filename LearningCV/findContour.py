import cv2
import numpy
img = cv2.imread("drawing.JPG")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,127,255,0)
# generate the binary image
contours,hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_TREE, # contour retrieval mode
    cv2.CHAIN_APPROX_SIMPLE # contour approximation method
)

img = cv2.drawContours(img,contours,-1,(0,255,0),3)

cv2.namedWindow("contours",cv2.WINDOW_NORMAL)
cv2.imshow("contours",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
