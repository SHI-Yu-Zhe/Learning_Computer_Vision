import cv2
cap = cv2.VideoCapture(0)
# capture video from device whose index is '0'

while(True):
    ret,frame = cap.read()
    # capture frame-by-frame
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # operations on frame:convert bgr to gray scale
    cv2.imshow('frame',gray)
    # display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
# when everything is done, release the capture
cv2.destroyAllWindows()