import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")
g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
g = np.float32(g)

corners = cv2.goodFeaturesToTrack(g,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),5,(0,255,0),-1)

plt.imshow(img)
plt.xticks([]),plt.yticks([])
plt.show()
