import numpy as np
import matplotlib.pyplot as plt

#from the class code

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


x=np.arange(-10, 10, 0.1)
y=np.exp(-0.5*x**2)
y_shifted = shift_conv(x, x.size//4)#//2

plt.title("Gaussian shifted by half the array length")
plt.plot(x,y, label="Original gaussian")
plt.plot(x,y_shifted, 'r', label="Shifted gaussian")
plt.legend()
plt.show()