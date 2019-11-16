import cv2
import numpy as np

img = cv2.imread("drawing.JPG")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours,hierarchy = cv2.findContours(thresh,1,2)

for i in range(len(contours)):
    cnt = contours[i]
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    pts = [leftmost,topmost,rightmost,bottommost]
    print(pts)
    for i in pts:
        center = i
        img = cv2.circle(img,i,5,(0,0,255),-1)

cv2.namedWindow("extreme points",cv2.WINDOW_NORMAL)
cv2.imshow("extreme points",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
