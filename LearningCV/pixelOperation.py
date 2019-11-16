import numpy as np
import cv2

img = cv2.imread("test.JPG")

px = img[100,100]
print(px)
# select some pixels of image
blue = img[100,100,0]
print(blue)
# access only blue pixels

# numpy array methods to access color scale
print(img.item(100,100,0)) # blue
print(img.item(100,100,1)) # green
print(img.item(100,100,2)) # red
