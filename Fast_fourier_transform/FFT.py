import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('download2.jpg',0)
# Perform FFT
f = np.fft.fft(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Apply threshold
magnitude_spectrum[magnitude_spectrum < 10] = 0

# Take inverse FFT
f_ishift = np.fft.ifftshift(magnitude_spectrum)
inverse_fft = np.fft.ifft(f_ishift)
img_back = np.real(inverse_fft)
mse = np.mean((img - img_back) ** 2)
psnr = 20*np.log10(255/np.sqrt(mse))


print(img_back)
print("now the value of PSNR")
print(psnr)
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()