#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 18:24:36 2022

@author: louis
"""

import numpy as np 
import matplotlib.pyplot as plt

def expo(x):
    return np.exp(x)
def expo_oo(x):
    return np.exp(0.01*x)

from scipy.misc import derivative

delta = 10**(-3.2)

x_expo = np.linspace(0,100, 1000)
y_expo = derivative(expo, x_expo, dx=delta)

plt.plot(x_expo, y_expo)
plt.plot(x_expo, expo(x_expo))
plt.show()