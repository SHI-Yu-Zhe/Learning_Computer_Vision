import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("test.JPG",0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))

while(True):
    cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    cv2.imshow("res", res)
    k = cv2.waitKey(1)
    if k == ord('q'):
        cv2.imwrite("testHE.JPG",equ)
        break
    elif k == 27:
        cv2.destroyAllWindows()
        break
