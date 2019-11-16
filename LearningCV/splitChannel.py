import numpy as np
import cv2

img = cv2.imread("test.JPG")
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
print(b,g,r)
color = [b,g,r]

for i in range(3):
    img = cv2.imread("test.JPG")
    img[:,:,i] = 0
    cv2.namedWindow("image"+str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("image"+str(i), img)

cv2.waitKey(0)
cv2.destroyAllWindows()
