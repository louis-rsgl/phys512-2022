#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:49:45 2022

@author: louis
"""
import numpy as np
f = open("/Users/louis/Desktop/McGill/FALL 2022/PHYS 512/Assignment/5/phys512-2022/Assignment5/planck chain.txt", "r")
def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]
lines = f.readlines()
chain = []
for line in lines:
    l = line.split(" ")
    l = remove_values_from_list(l, "")
    l = l[1:7]
    my_line = []
    for i in range(0,6):
        my_line.append(float(l[i]))
    chain.append(np.array(my_line))
chain = np.array(chain)

nchain = 6
means=np.zeros([nchain,chain.shape[1]])
scats=np.zeros([nchain,chain.shape[1]])
for i in range(nchain):
    means[i,:]=np.mean(chain[i:],axis=0)
    scats[i,:]=np.std(chain[i:],axis=0)
#scatter of means
mean_scat=np.std(means,axis=0)
gelman_rubin=mean_scat/np.mean(scats,axis=0)
print('gelman_rubin scatters are ',gelman_rubin)