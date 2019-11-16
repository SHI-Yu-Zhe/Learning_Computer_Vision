import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("drawing.JPG")
g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


