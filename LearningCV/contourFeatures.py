import cv2
import numpy
img = cv2.imread("drawing.JPG")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,127,255,0)
contours,hierarchy = cv2.findContours(thresh,1,2)

cnt = contours[0]
M = cv2.moments(cnt)
# now get the moment matrix

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
# calculate x,y ordinates of centroid

area = int(M['m00'])
# calculate area of contour shape

perimeter = cv2.arcLength(cnt,True)
# calculate perimeter of contour shape

hull = cv2.convexHull(cnt)
# calculate convex hull of contours

k = cv2.isContourConvex(cnt)
# judge is contour a convex hull

print("centroid:("+str(cx)+","+str(cy)+")")
print("area:"+str(area))
print("perimeter:"+str(perimeter))
print("is contour convex:"+str(k))

