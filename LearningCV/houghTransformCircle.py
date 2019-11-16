import cv2
import numpy as np

img = cv2.imread("drawing.JPG")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,5)

circles = cv2.HoughCircles(
    gray,cv2.HOUGH_GRADIENT,1,20,
    param1=50,param2=30,
    minRadius=0,maxRadius=0
)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),4)
    cv2.circle(img,(i[0],i[1]),2,(255,255,0),5)
while(True):
    cv2.namedWindow("res",cv2.WINDOW_NORMAL)
    cv2.imshow("res",img)
    k = cv2.waitKey(1)
    if k == ord("q"):
        cv2.destroyAllWindows()
        break
