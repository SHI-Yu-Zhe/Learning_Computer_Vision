import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("test.JPG",0)
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(
    dft_shift[:,:,0],
    dft_shift[:,:,1]
))

row,col = img.shape
rowc,colc = int(row/2),int(col/2)
mask = np.zeros((row,col,2),np.uint8)
mask[rowc-30:rowc+30,colc-30:colc+30] = 1
# create a mask with its centre white

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img,cmap='gray')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(img_back,cmap='gray')
plt.xticks([]),plt.yticks([])
plt.show()
