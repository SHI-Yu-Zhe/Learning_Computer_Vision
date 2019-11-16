import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("test.JPG")

b = [255,0,0]

replicate = cv2.copyMakeBorder(
    img,
    500,500,500,500,
    cv2.BORDER_REPLICATE
)
reflect = cv2.copyMakeBorder(
    img,
    500,500,500,500,
    cv2.BORDER_REFLECT
)
reflect101 = cv2.copyMakeBorder(
    img,
    500,500,500,500,
    cv2.BORDER_REFLECT101
)
wrap = cv2.copyMakeBorder(
    img,
    500,500,500,500,
    cv2.BORDER_WRAP
)
constant = cv2.copyMakeBorder(
    img,
    500,500,500,500,
    cv2.BORDER_CONSTANT,
    value = (0,255,0)
)
title = ['original','replicate','reflect','reflect101','wrap','constant']
image = [img,replicate,reflect,reflect101,wrap,constant]
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(image[i],'gray')
    plt.title(str.upper(title[i]))
    plt.xticks([]),plt.yticks([])
plt.show()
