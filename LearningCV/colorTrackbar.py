import cv2
import numpy as np

def nothing(x):
    pass
img = np.zeros((300,512,3),np.uint8)
cv2.namedWindow("trackbar")
# define a basic black image and a window

cv2.createTrackbar("R","trackbar",0,255,nothing)
cv2.createTrackbar("G","trackbar",0,255,nothing)
cv2.createTrackbar("B","trackbar",0,255,nothing)
# create r,g,b track bars

switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch,"trackbar",0,1,nothing)
# create a switch for ON/OFF functionality

while(True):
    cv2.imshow("trackbar",img)
    k = cv2.waitKey(1)
    if k == 27:
        break

    r = cv2.getTrackbarPos('R','trackbar')
    g = cv2.getTrackbarPos('G','trackbar')
    b = cv2.getTrackbarPos('B','trackbar')
    s = cv2.getTrackbarPos(switch,'trackbar')
    # get position of each track bar

    color = np.uint8([[[b,g,r]]])
    hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    print(hsv_color)
    # get hsv of sselected color

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]
    # if switched to OFF, just print original black image
    # else, print bgr color image

cv2.destroyAllWindows()
