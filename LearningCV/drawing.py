import cv2
import numpy as np
img = np.zeros((512,512,3),np.uint8)
# create a black image
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
# create a diagonal blue line
img = cv2.rectangle(img,(384,0),(512,128),(0,255,0),3)
# create a green rectangle at the top-right corner
img = cv2.circle(img,(447,63),63,(0,0,255),-1)
# create a red-filled circle in the rectangle above
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# create a half ellipse at the centre of image
pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
# input vertices as an array, in int 32 bits
pts = pts.reshape(-1,1,2)
# reshape the array into a rows*1*2 array
img = cv2.polylines(img,[pts],True,(0,255,255))
# create a closed polygon with 4 vertices in yellow
font = cv2.FONT_HERSHEY_COMPLEX
# select font style
img = cv2.putText(
    img,
    "OpenCV", # text
    (10,500), # position coordinates
    font, # font style
    4, # font scale
    (255,255,255),
    2, # thickness
    cv2.LINE_AA # to make lines prettier
)
cv2.namedWindow("drawing",cv2.WINDOW_NORMAL)
cv2.imshow("drawing",img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("drawing.JPG",img)
    cv2.destroyAllWindows()
