import cv2
import numpy as np

# read the image
img = cv2.imread("test.JPG",0)
cv2.namedWindow("test",cv2.WINDOW_NORMAL)
cv2.imshow("test",img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("testgray.JPG",img)
    cv2.destroyAllWindows()