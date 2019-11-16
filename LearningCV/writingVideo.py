import cv2
cap = cv2.VideoCapture("test.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    "graytest.avi",
    fourcc,
    20.0,
    (640,480)
)
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)
        # vertical direction by default
        out.write(frame)
        # write the flipped frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
