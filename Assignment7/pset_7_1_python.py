from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random as rd





###Data making with random
"""
n=30253
x_y_z_new =np.empty(n*3,dtype='int32')
rd.seed(1)
for i in range (n*3):
    x_y_z_new[i] = rd.randint(0, 1e8)

vv=np.reshape(x_y_z_new,[n,3])
maxval=1e8
vmax=np.max(vv,axis=1)
vv2=vv[vmax<maxval,:]
x = []
y = []
z = []


f=open('./Assignment7/rand_python_points.txt','a')
for i in range(vv2.shape[0]):
    myline=repr(vv2[i,0])+' '+repr(vv2[i,1])+' '+ repr(vv2[i,2])+'\n'
    f.write(myline)
    x.append(vv2[i,0])
    y.append(vv2[i,1])
    z.append(vv2[i,2])
f.close()
"""

### We get the data we made and plot it in 3D
f=open('./Assignment7/rand_python_points.txt','r')

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


plt.plot(0.16*x - 1.2*y, z,".",markersize=1)
plt.xlabel("-0.1x+0.05y")
plt.ylabel("z")
plt.title("rand_python_points.txt points in the (ax + by, z) plane")
plt.show()
