#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 18:24:36 2022

@author: louis
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy import interpolate

dat=np.loadtxt('lakeshore.txt')

def lakeshore(V,data):
    V_points = []
    T_points = []
    for line in data:
        V_points.append(line[0])
        T_points.append(line[1])

    tck = interpolate.splrep(V_points, T_points)
    True_value = 0
    for line_index in range(len(data)):
        if V_points[line_index] == data[line_index][1]:
            True_value = line_index
    return str(interpolate.splev(V, tck)) +  " +/- " + str(np.std(interpolate.splev(V, tck)-True_value))

print(lakeshore(0.182832, dat))


