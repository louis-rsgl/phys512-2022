import numpy as np
from matplotlib import pyplot as plt

def logistic(x,y):
    dydx=y/(1+x**2) ##ODE
    return dydx


def rk4_step(fun,x,y,h): ##classe step function
    k1=fun(x,y)*h
    k2=h*fun(x+h,y+k1)
    k3=h*fun(x+h,y+k2)
    k4=h*fun(x+h,y+k3)
    k5=h*fun(x+h,y+k4)
    k6=h*fun(x+h,y+k5)
    dy=(k1+2*k2+2*k3+k4+k5+k6)/6
    return y+dy


npt=201
x=np.linspace(-20,20,npt)#ODE solver from -20 to 20
y=np.zeros([2,npt])
y[0,0]=1
y[1,0]=0
for i in range(npt-1):
    h=x[i+1]-x[i]
    y[:,i+1]=rk4_step(logistic,x[i],y[:,i],h)


def rk4_stepd(fun,x,y,h):
    length_h_step = rk4_step(fun,x,y,h) # actual step
    length_h2_step = rk4_step(fun,x,y,h/2) # first half step
    length_h22_step = rk4_step(fun,x+ h/2,length_h2_step,h/2) # 2nd half step
    return (16*length_h22_step - length_h2_step)/15 # from numerical recipees