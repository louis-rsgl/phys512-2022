import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt



half_life_list= [
    4.468 * 10**9 * np.pi*10**7,
    24.10 * 24 * 3600,
    6.70 * 3600,
    245500 * np.pi* 10**7,
    75380 * np.pi* 10**7,
    1600 * np.pi* 10**7,
    3.8235 * 24 * 3600,
    3.10 * 60,
    26.8 * 60,
    19.9 * 60,
    164.3 * 10**-6,
    22.3 * np.pi*10**7,
    5.015 * np.pi*10**7,
    138.376 * np.pi*10**7,
    0
]


def fun(time_spent,y,lifetime=half_life_list):
    dydtime=np.zeros(len(lifetime)+1)
    for i in range(len(lifetime)):
        if i == 0:
            dydtime[i] = -y[i]/lifetime[i]#Botemane equations
        elif i == len(lifetime):
            dydtime[i] = y[i-1]/lifetime[i-1]
        else:
            dydtime[i] = y[i-1]/lifetime[i-1] - y[i]/lifetime[i]

    return dydtime

initial_material = [1]
for i in range(15):
    initial_material.append(0)
initial_material = np.array(initial_material)
decay_pure_238=integrate.solve_ivp(fun,(0, half_life_list[0]),initial_material,method='Radau',interval=np.linspace(0, half_life_list[0],1000))
print(decay_pure_238)
    




