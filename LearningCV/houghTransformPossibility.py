import cv2
import numpy as np

img = cv2.imread("drawing.JPG")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,150,250,apertureSize=3)

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=1,maxLineGap=1)
for x1,x2,y1,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),4)
while(True):
    cv2.namedWindow("res",cv2.WINDOW_NORMAL)
    cv2.imshow("res",img)
    k = cv2.waitKey(1)
    if k == ord("q"):
        cv2.destroyAllWindows()
        break
