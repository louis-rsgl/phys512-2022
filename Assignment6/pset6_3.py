import numpy as np
import matplotlib.pyplot as plt

#from class notes
def correlation(f,g):
    ft1 = np.fft.fft(f)
    ft2 = np.conj(np.fft.fft(g))
    return np.real(np.fft.ifft(ft1*ft2))

def correlation_fixed(arr, arr1):
    for i in range(arr.size):
        arr = np.append(arr, np.array([0]))
        arr1 =np.append(arr1, np.array([0]))
    return correlation(arr, arr1)
def shift_conv(array, amount):
    y=np.exp(-0.5*array**2)
    N=array.size
    J=complex(0,1)
    yft=np.fft.fft(y)
    kvec=np.arange(N)
    dx=amount
    yft_shifted=yft*np.exp(-2*np.pi*J*kvec*dx/N)
    y_shifted=np.real(np.fft.ifft(yft_shifted))
    return y_shifted

arr=np.arange(-10,10,0.1)
gauss=np.exp(-0.5*arr**2)


gauss_shifted = shift_conv(arr, arr.size//4)#//2


gauss_corr_itself=correlation_fixed(gauss_shifted,gauss_shifted)
plt.title("Shifted Gaussian correlated with itself")

plt.plot(arr, gauss_shifted, label="Original shifted gaussian")
arr=np.arange(-10,10,0.05)
plt.plot(arr, np.fft.fftshift(gauss_corr_itself), label="Correlated shifted gaussian")
plt.legend()
plt.show()

