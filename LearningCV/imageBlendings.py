import cv2
import numpy as np,sys
A = cv2.imread("test.JPG")
B = cv2.imread("testgray.JPG")

# generate gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate laplacian pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate laplacian pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack(A[:,:cols/2],B[:,cols/2:])
    LS.append(ls)

# reconstruct original image
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_,LS[i])

cv2.namedWindow("res",ls_)
cv2.imshow("res",ls_)
k = cv2.waitKey(0)
while(True):
    if k == 27:
        cv2.destroyAllWindows()
        break
    elif k == ord('q'):
        cv2.imwrite("blending.JPG",ls_)
        cv2.destroyAllWindows()
        break
