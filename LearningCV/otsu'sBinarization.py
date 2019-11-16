import cv2
import numpy as np
img = cv2.imread("test.JPG",0)
blur = cv2.GaussianBlur(img,(5,5),0)

hist = cv2.calcHist([blur],[0],None,[256],[0,256])
# find normalized histogram, and its cumulative distribution function
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
thresh = -1

for i in range(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of class
    b1,b2 = np.hsplit(bins,[i]) # weights

    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1,np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,\
            np.sum(((b2-m2)**2)*p2)/q2

    # minimization function
    fn = q1*v1+q2*v2
    if fn < fn_min:
        fn_min = thresh = i

# find otsu's threshold value
ret,otsu = cv2.threshold(
    blur,
    0,255,
    cv2.THRESH_BINARY+cv2.THRESH_OTSU
)
print(thresh,ret)