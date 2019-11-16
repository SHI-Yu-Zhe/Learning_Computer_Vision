import cv2
import numpy as np
img = cv2.imread("shi.JPG")
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
res1 = clahe.apply(b)
res2 = clahe.apply(g)
res3 = clahe.apply(r)
res = cv2.add(res1,res2)
res = cv2.add(res,res3)
while(True):
    cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    cv2.imshow("res", res)
    k = cv2.waitKey(1)
    if k == ord("q"):
        cv2.imwrite("shires.JPG",res)
        break
    elif k == 27:
        cv2.destroyAllWindows()
        break
