import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    _,frame = cap.read()
    # take each frame

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # convert each frame from bgr to hsv

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # define range of blue in hsv

    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    # threshold the hsv image to get only blue

    res = cv2.bitwise_and(frame,frame,mask=mask)
    # bitwise-and mask and original image to capture blue object

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('result',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()