import numpy as np
import numba as nb
from matplotlib import pyplot as plt
from scipy import fft
from scipy.spatial import distance_matrix
from scipy.signal import convolve2d
import math


G = 6.67e-11
@nb.njit(parallel=True)
def set_grad(xy,pot,grad):
    #n=xy.shape[0]
    n=pot.shape[0]
    for i in nb.prange(xy.shape[0]):
        if xy[i,0]<0:
            ix0=n-1
            ix1=0
            fx=xy[i,0]+1
        else:
            ix0=int(xy[i,0])
            ix1=ix0+1
            fx=xy[i,0]-ix0
            if ix1==n:
                ix1=0
        if xy[i,1]<0:
            iy0=n-1
            iy1=0
            fy=xy[i,1]+1
        else:
            iy0=int(xy[i,1])
            iy1=iy0+1
            fy=xy[i,1]-iy0
            if iy1==n:
                iy1=0
        grad[i,0]=(pot[ix1,iy1]-pot[ix0,iy1])*fy+(pot[ix1,iy0]-pot[ix0,iy0])*(1-fy)
        grad[i,1]=(pot[ix1,iy1]-pot[ix1,iy0])*fx+(pot[ix0,iy1]-pot[ix0,iy0])*(1-fx)
class particles:
    def __init__(self,number_of_particles, grid_size):
        self.grid_size = grid_size
        self.number_of_particles = number_of_particles

        self.gradients = np.zeros([self.number_of_particles,2])
        self.coordinates=np.zeros([number_of_particles,2])
        self.masses=np.ones(number_of_particles)
        self.forces = np.zeros([number_of_particles,2])
        self.velocities = np.zeros([number_of_particles,2])
        self.one_particle_pot = np.zeros([self.grid_size,self.grid_size])
        self.densities=np.zeros([self.grid_size,self.grid_size])
        self.potentials=np.zeros([self.grid_size,self.grid_size])
    
    def closest_cell(self):
        new_coor = np.transpose(parts.coordinates)
        for i in range(2):
            for j in range(len(new_coor[0])):
                new_coor[i][j] = math.floor(new_coor[i][j])+0.5
    
    def set_density(self):
        self.closest_cell()
        rho = self.densities
        coor = parts.coordinates
        for i in range(self.number_of_particles):
            x = int(coor[i][0] - 0.5)
            y = int(coor[i][1] - 0.5)
            rho[x][y] = rho[x][y] + 1
    
    def set_potential(self):
        self.potentials=fft.irfft2(fft.rfft2(self.densities)*fft.rfft2(self.one_particle_pot),[self.grid_size,self.grid_size])
    def set_one_particle_pot(self):
        distance = []
        for i in range(self.grid_size):
            dist = []
            for j in range(self.grid_size):
                dist.append((i**2 + j**2)/(2*self.grid_size**2))
            distance.append(dist)
        distance[0][0] = 1
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                distance[i][j] = G / distance[i][j]
        distance[0][0] = 0
        self.one_particle_pot = distance
    def take_step(self,dt):
        self.coordinates=self.coordinates+dt*self.velocities
        self.set_potential()
        self.set_forces()
        self.velocities=self.velocities+self.forces*dt
    
    def set_forces(self):
        set_grad(self.coordinates, self.potentials,self.gradients)
        self.forces=self.gradients

num_part = 10
grid_size = 3
parts=particles(num_part, grid_size)
x = np.random.rand(num_part)*grid_size
y = np.random.rand(num_part)*(-grid_size)
parts.coordinates = np.array([x, y]).transpose()# set random numbers within the grid (grid_size*grid_size)
#parts.set_one_particle_pot()
#parts.set_density()
#parts.set_potential()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_xlim(-2, 2)
data = np.linspace(0, N, N)
ax.set_xticks()
ax.set_ylim(-2, 2)
ax.plot(x, y, '.')
plt.show()
"""
fig = plt.figure()
for i in range(3000):
    plt.plot(np.transpose(parts.coordinates)[0],np.transpose(parts.coordinates)[1], '.')
    plt.pause(0.0001)
    parts.take_step(dt=100)

xy=parts.coordinates.copy()
parts.set_potential()
rho=parts.densities.copy()
pot=parts.potentials.copy()

fig = plt.figure()
ax = fig.add_subplot(111)
crap=ax.imshow(parts.densities**0.5)

for i in range(1500):
    
    for j in range(3):
        parts.take_step(dt=1)
    
    crap.set_data(parts.densities**0.5)
    plt.pause(0.001)

fig = plt.figure()
ax = fig.add_subplot(111)
crap=ax.imshow(parts.densities**0.5)

for i in range(1500):
    
    parts.take_step(dt=100)
    crap.set_data(parts.densities**0.5)
    plt.pause(0.001)"""