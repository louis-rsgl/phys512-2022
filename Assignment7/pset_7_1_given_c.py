import matplotlib.pyplot as plt
import numpy as np
import random as rd


###### We get the data from rand_points.txt and plot it in 3D

f = open("/Users/louis/Desktop/McGill/FALL 2022/PHYS 512/Assignment/7/phys512-2022/Assignment7/rand_points.txt", 'r')

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
plt.title("rand_points.txt points in the (ax + by, z) plane")
plt.show()


