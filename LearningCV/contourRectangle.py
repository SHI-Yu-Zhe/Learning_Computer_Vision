import cv2
import numpy
img = cv2.imread("drawing.JPG")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours,hierarchy = cv2.findContours(thresh,1,2)
for i in range(len(contours)):
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    # x,y ordinates of rectangle; width and height of rectangle
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
# loop to draw rectangle grids over every object
cv2.namedWindow("res",cv2.WINDOW_NORMAL)
cv2.imshow("res",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
