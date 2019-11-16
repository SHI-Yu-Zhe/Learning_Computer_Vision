import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("drawing.JPG",0)
img2 = img.copy()
template = img[200:400,200:400]
w,h = template.shape[::-1]

methods = [
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED
]
titles = [
    'TM_CCOEFF',
    'TM_CCOEFF_NORMED',
    'TM_CCORR',
    'TM_CCORR_NORMED',
    'TM_SQDIFF',
    'TM_SQDIFF_NORMED'
]
for i in range(len(methods)):
    img = img2.copy()

    res = cv2.matchTemplate(img,template,methods[i])
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

    # two cases to take minimum instead of maximum
    if i in [cv2.TM_SQDIFF_NORMED,cv2.TM_SQDIFF]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0]+w,top_left[1]+h)

    cv2.rectangle(img,top_left,bottom_right,255,2)

    plt.subplot(121)
    plt.imshow(res,cmap='gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.subplot(122)
    plt.imshow(img,cmap='gray')
    plt.title("detected point")
    plt.xticks([]),plt.yticks([])
    plt.show()
