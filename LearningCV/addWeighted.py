import cv2

img1 = cv2.imread("test.JPG")
img1 = img1[0:500,0:500]
img2 = cv2.imread("drawing.JPG")
img2 = img2[0:500,0:500]


dst = cv2.addWeighted(img1,0.4,img2,0.6,0)

cv2.namedWindow('destination',cv2.WINDOW_NORMAL)
cv2.imshow('destination',dst)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("blending.JPG",dst)
    cv2.destroyAllWindows()

