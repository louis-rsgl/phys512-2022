#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 18:24:36 2022

@author: louis
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.misc import derivative
 
def ndiff(fun, x, full=False):
    e = 10**(-16) #Here is the 
    dx = 10**(-8)
    if full:
        derivative_estimate = (fun(x + dx) - fun(x - dx))/(2*dx)
        error = derivative(fun, x, dx=dx) - derivative_estimate
        return derivative_estimate, dx, error
    else:
        derivative_estimate = (fun(x + dx) - fun(x - dx))/(2*dx)
        return derivative_estimate
        
def fun(x):
    return x**2
print(ndiff(fun, 5, True))
