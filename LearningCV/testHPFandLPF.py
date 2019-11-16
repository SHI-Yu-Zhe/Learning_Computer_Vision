import cv2
import numpy as np
from matplotlib import  pyplot as plt

mean_filter = np.ones((3,3))

# gaussian filter
x = cv2.getGaussianKernel(10,5)
gaussian = x.T

# edge detecting filters
# scharr in x direction
scharr = np.array([[-3,0,3],
                  [-10,0,10],
                  [-3,0,3]])

# sobel in y direction
sobel_y = np.array([[-1,-2,-1],
                   [0,0,0],
                   [-1,-2,-1]])

# sobel in x direction
sobel_x = np.array([[-1,0,-1],
                   [-2,0,-2],
                   [-1,0,-1]])

# laplacian filter
laplacian = np.array([[0,1,0],
                     [1,-4,1],
                     [0,1,0]])

filters = [mean_filter,gaussian,scharr,sobel_x,sobel_y,laplacian]
names = ['mean_filter','gaussian','scharr','sobel_x','sobel_y','laplacian']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
magnitude_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

for i in range(len(filters)):
    plt.subplot(2,3,i+1)
    plt.imshow(magnitude_spectrum[i],cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.title(names[i])
plt.show()
