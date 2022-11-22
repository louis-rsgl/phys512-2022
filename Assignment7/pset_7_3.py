#code from class

import numpy as np
from matplotlib import pyplot as plt

u=np.linspace(0,1,2001)
u=u[1:]
v=2*u*(-np.log(u))
plt.figure(figsize=(18,9))
plt.title("Droplet")
plt.plot(u, np.zeros(2000)+2/np.e, label="Max v = 2/e")
plt.plot(u,v, label="Droplet")
plt.xlabel("u")
plt.ylabel("v")
plt.legend()
plt.show()

N=1000000
u=np.random.rand(N)
v=np.random.rand(N)*0.73
r=v/u
exponential_accepted=np.where(v<= 2*u*(-np.log(u)))
bins=np.linspace(1,6,501)
plt.figure(figsize=(18,9))
n, bin, patches = plt.hist(r[exponential_accepted], bins, label="Histogram of the distribution")
pred=np.exp(-bins)*n.max()*np.e
plt.plot(bins,pred, label="predicted exponential deviate")
plt.title("Histogram of the ratio of uniforms method")
plt.legend()
plt.show()