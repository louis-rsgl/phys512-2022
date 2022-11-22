import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


def power_law_cdf(x, alpha):
    return x**(1/(1-alpha))
def power_law(x, alpha):
    return x**(-alpha)
def exp_ (x):
    return np.e**(-x)
x = np.random.rand(1000000)
s = power_law_cdf(x, 2.7)
bins=np.linspace(1,6,501)

plt.figure(figsize=(18,9))
plt.title("Histogram of the powerlaw distribution that matched the power law ")
n, bin, patches = plt.hist(s, bins, label="Histogram of the random distribution")
plt.plot(bins, n.max()*power_law(bins, 2.7), label="Power law")
plt.legend()
plt.show()
s_x = power_law_cdf(np.random.rand(10000), 2.7)
random_power_law_set = s_x[s_x<=10]

random_values_under_exp = []
random_values_over_exp = []
random_values = np.random.rand(len(random_power_law_set))
exp_random = exp_(random_power_law_set)
power_random = power_law(random_power_law_set, 2.7)
index_accepted = np.where(random_values * power_random<= exp_random)

random_values_under_exp =  exp_random[index_accepted]
plt.figure(figsize=(18,9))
plt.title("Rejection method with powerlaw bound on exponetial deviate")
plt.plot(random_power_law_set, random_values * power_random, '.', markersize=1.3, label="rejected")
plt.plot(random_power_law_set[index_accepted], (random_values * power_random)[index_accepted], '.', markersize=1.3, label="rejected")
plt.plot(random_power_law_set, power_random, '.', markersize=.3, label="Powerlaw with alpha = 2.7" )
plt.plot(random_power_law_set, exp_random, '.', markersize=.3, label="Exponential deviate" )
plt.legend()
plt.show()


percentage = len(index_accepted[0])*100/random_values.size
print("The precentage of accepted data is " + str(percentage) + "%")