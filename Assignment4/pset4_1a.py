import numpy as np
from matplotlib import pyplot as plt


data=np.load('Assignment4/sidebands.npz')
t=data['time']
d_true=data['signal']

t0_init = t[np.where(d_true == d_true.max())][0]
w_init = 0.0000774/4
a_init = d_true.max()

p0=np.array([a_init , w_init , t0_init])

def calc_lorentz(p, t):
    a = p[0]
    w = p[1]
    t0 = p[2]
    y = a/(1 + ((t-t0)/w)**2)
    grad=np.zeros([t.size,p.size])
    #now differentiate w.r.t. all the parameters
    grad[:,0]=1/(1 + ((t-t0)/w)**2)
    grad[:,1]=2 * a * (t - t0)**2 / (w**3 * (1+ ((t - t0) / w)**2)**2)
    grad[:,2]=2 * a * (t - t0) / (w**2 * (1 + ((t - t0) / w)**2)**2)

    return y, grad
d_pred_init, grad = calc_lorentz(p0, t)
plt.figure(figsize=(16,9))

plt.plot(t,d_true, label="Our data")
plt.plot(t, d_pred_init, label="Initial step")

p=p0.copy()
for j in range(100):
    pred,grad=calc_lorentz(p,t)
    r=d_true-pred
    err=(r**2).sum()
    r=np.matrix(r).T
    grad=np.matrix(grad)

    lhs=grad.T @ grad
    rhs=grad.T @ r
    dp=np.linalg.inv(lhs) @ rhs
    for jj in range(p.size):
        p[jj]=p[jj]+dp[jj]

noise = np.std(np.abs(pred - d_true))
N_1 = noise**(-2)*np.identity(len(p))
covariance_expectation_value = np.linalg.inv( grad.T @ grad @ N_1 )
p_err = np.sqrt(np.diag(covariance_expectation_value))


print("a is " + str(p[0]) + " ± " + str(p_err[0]))
print("w is " + str(p[1]) + " ± " + str(p_err[1]))
print("t0 is " + str(p[2]) + " ± " + str(p_err[2]))


print("error on fit is " +  str(np.std(np.abs(pred - d_true))))
plt.plot(t, pred, label="Predicted data")
plt.xlabel("t")
plt.ylabel("d")
plt.legend()
plt.show()