import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
dish = open("./dish_zenith.txt", "r")

x = []
y = []
z = []
for i in dish.readlines():
    splitter = i.split()
    x.append(float(splitter[0]))
    y.append(float(splitter[1]))
    z.append(float(splitter[2]))



X, Y = np.meshgrid(x, y)

def dish_function(x, y, x0, y0, z0, a):
    return a*x**2 -2*a*x*x0 + a*x0**2 + a*y**2 - 2*a*y*y0 + a*y0**2 + z0

def _dish_function(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//5):
       arr += dish_function(x, y, *args[i*5:i*5+5])
    return arr

Z = np.zeros(X.shape)
for p in gprms:
    Z += dish_function(X, Y, *p)
Z += noise_sigma * np.random.randn(*Z.shape)
xdata = np.vstack((X.ravel(), Y.ravel()))

popt, pcov = curve_fit(_dish_function, xdata, Z.ravel(), p0)

print(fit)