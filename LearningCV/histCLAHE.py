import cv2
import numpy as np
img = cv2.imread("test.JPG",0)
res1 = cv2.imread("testHE.JPG",0)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
# create clahe object
res2 = clahe.apply(img)
res = np.hstack((res1,res2))
while(True):
    cv2.namedWindow("res",cv2.WINDOW_NORMAL)
    cv2.imshow("res",res)
    k = cv2.waitKey(1)
    if k == ord('q'):
        cv2.imwrite("testCLAHE.JPG",res2)
        break
    elif k == 27:
        cv2.destroyAllWindows()
        break
