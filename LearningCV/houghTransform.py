import cv2
import numpy as np

img = cv2.imread("drawing.JPG")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize=3)
# apertureSize should be odd between 3 and 7

lines = cv2.HoughLines(gray,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho*a
    y0 = rho*b
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*a)
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*a)

    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),4)

while(True):
    cv2.namedWindow("res",cv2.WINDOW_NORMAL)
    cv2.imshow("res",img)
    k = cv2.waitKey(1)
    if k == ord("q"):
        cv2.destroyAllWindows()
        break
