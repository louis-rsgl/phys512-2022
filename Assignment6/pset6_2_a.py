import numpy as np
import matplotlib.pyplot as plt

#from class notes
def correlation(f,g):
    ft1 = np.fft.rfft(f)
    ft2 = np.conj(np.fft.rfft(g))
    return np.fft.irfft(ft1*ft2)



arr=np.arange(-10,10,0.1)
gauss=np.exp(-0.5*arr**2)

gauss_corr_itself=correlation(gauss,gauss)



plt.title("Gaussian correlated with itself")

plt.plot(arr, gauss, label="Original gaussian")
plt.plot(arr, np.fft.fftshift(gauss_corr_itself), label="Correlated gaussian")
plt.legend()
plt.show()

