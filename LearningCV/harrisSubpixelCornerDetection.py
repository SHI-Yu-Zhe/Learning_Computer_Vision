import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")
g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
g = np.float32(g)

dst = cv2.cornerHarris(g,2,3,0.04)
dst = cv2.dilate(dst,None)
ret,dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

ret,labels,states,centroids = cv2.connectedComponentsWithStats(dst)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.01)
corners = cv2.cornerSubPix(g,np.float32(centroids),(5,5),(-1,-1),criteria)

res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]] = [0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

plt.imshow(img)
plt.show()
