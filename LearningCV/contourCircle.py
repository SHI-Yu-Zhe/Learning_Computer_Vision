import cv2
import numpy
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()
contours,hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_NONE
)
for i in range(len(contours)):
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    img3 = cv2.rectangle(img3,(x,y),(x+w,y+h),(0,255,0),3)
    # add rectangle contours of every object
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    # get min enclosing circle of object
    center = (int(x),int(y))
    radius = int(radius)
    img = cv2.circle(img,center,radius,(0,255,0),3)
    # add circle contours of every object
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        img1 = cv2.ellipse(img1,ellipse,(0,255,0),3)
    # add ellipse contours of every object
    [vx,vy,x,y] = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
    img2 = cv2.line(img2,(x-w/2,y-h/2),(x+w/2,y+h/2),(0,255,0),3)
    # add line fitting of every object
res = [img3,img,img1,img2]
title = ['rectangle','circle','ellipse','line-fitting']
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(res[i])
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])
plt.show()

