import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)
# state initialization
drawing = False
# true if the mouse is pressed
mode = True
# if true, draw rectangle.press 'm' to toggle to curve
ix,iy = -1,-1
# initialize original coordinates
# define mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy=x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(255,255,0),-1)
            else:
                cv2.circle(img,(x,y),5,(255,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(255,255,0),-1)
        else:
            cv2.circle(img,(x,y),1,(255,255,0),-1)

cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image",draw_circle)

while(True):
    cv2.imshow("image",img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()