import numpy as np
import numba as nb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random as rd


###Data making with C
"""
mylib=ctypes.cdll.LoadLibrary('libc.dylib')

rand=mylib.rand
rand.argtypes=[]
rand.restype=ctypes.c_int


@nb.njit
def get_rands_nb(vals):
    n=len(vals)
    for i in range(n):
        vals[i]=rand()
    return vals

def get_rands(n):
    vec=np.empty(n,dtype='int32')
    get_rands_nb(vec)
    return vec


n=300000000
vec=get_rands(n*3)
#vv=vec&(2**16-1)

vv=np.reshape(vec,[n,3])
vmax=np.max(vv,axis=1)

maxval=1e8
vv2=vv[vmax<maxval,:]

f=open('./Assignment7/our_rand_points.txt','a')
for i in range(vv2.shape[0]):
    myline=repr(vv2[i,0])+' '+repr(vv2[i,1])+' '+ repr(vv2[i,2])+'\n'
    f.write(myline)
f.close()
"""

###### We get the data from rand_points.txt and plot it in 3D

f = open("./Assignment7/our_rand_points.txt", 'r')

xyz = f.readlines()

x = []
y = []
z = []

for line in xyz:
    xyzn = line.split("\n")
    x_y_z = xyzn[0].split(" ")
    x.append(int(x_y_z[0]))
    y.append(int(x_y_z[1]))
    z.append(int(x_y_z[2]))

x = np.array(x)
y = np.array(y)
z = np.array(z)


###We plot the data in (ax + by, z) plane
fig = plt.figure(figsize=(16,8))


plt.plot(-0.1*x + 0.05*y, z,".",markersize=1)
plt.xlabel("-0.1x+0.05y")
plt.ylabel("z")
plt.title("our_rand_points.txt points in the (ax + by, z) plane")
plt.show()


