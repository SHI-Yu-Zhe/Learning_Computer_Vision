import cv2
img = cv2.imread('test.JPG')
img = cv2.line(
    img,
    (0,0),(1000,1000),
    (0,255,0),
    10
)
cv2.namedWindow('line',cv2.WINDOW_NORMAL)
cv2.imshow('line',img)
k = cv2.waitKey(0)
if k== 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('testline.JPG', img)
    cv2.destroyAllWindows()
