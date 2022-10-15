import numpy as np
from matplotlib import pyplot as plt


data=np.load('Assignment4/sidebands.npz')
t=data['time']
d_true=data['signal']

t0_init = t[np.where(d_true == d_true.max())][0]
w_init = 0.0000774/4
a_init = d_true.max()
b_init = 0
c_init =0
dt_init = 0

p0=np.array([a_init , w_init , t0_init, b_init, c_init, dt_init])

def calc_lorentz(p, t):
    a = p[0]
    w = p[1]
    t0 = p[2]
    b = p[3]
    c = p[4]
    dt = p[5]
    y = (a/(1 + ((t-t0)/w)**2)) +  (b/(1 + ((t-t0+dt)/w)**2)) + (c/(1 + ((t-t0-dt)/w)**2))
    grad=np.zeros([t.size,p.size])
    #now differentiate w.r.t. all the parameters
    h = 1e-6
    p1_a = (a + h)/(1 + ((t-t0)/w)**2)
    p2_a = (a - h)/(1 + ((t-t0)/w)**2)
    da = (p1_a - p2_a) / (2 * h) 
    
    p1_w = (a /(1 + ((t-t0)/(w + h))**2)) +  (b/(1 + ((t-t0+dt)/(w + h))**2)) + (c/(1 + ((t-t0-dt)/(w + h))**2))
    p2_w = (a /(1 + ((t-t0)/(w - h))**2)) +  (b/(1 + ((t-t0+dt)/(w - h))**2)) + (c/(1 + ((t-t0-dt)/(w - h))**2))
    dw = (p1_w - p2_w) / (2 * h)
    
    p1_t0 = (a /(1 + ((t-(t0 + h))/w)**2)) +  (b/(1 + ((t-(t0 + h)+dt)/w)**2)) + (c/(1 + ((t-(t0 + h)-dt)/w)**2))
    p2_t0 = (a /(1 + ((t-(t0 - h))/w)**2)) +  (b/(1 + ((t-(t0 - h)+dt)/w)**2)) + (c/(1 + ((t-(t0 - h)-dt)/w)**2))
    dt0 =  (p1_t0 - p2_t0) / (2 * h) 
    
    p1_b = (b + h)/(1 + ((t-t0+dt)/w)**2)
    p2_b = (b - h)/(1 + ((t-t0+dt)/w)**2)
    db = (p1_b - p2_b) / (2 * h) 
    
    p1_c = (c + h)/(1 + ((t-t0-dt)/w)**2)
    p2_c = (c - h)/(1 + ((t-t0-dt)/w)**2)
    dc = (p1_c - p2_c) / (2 * h) 
    
    p1_dt = (a /(1 + ((t-t0)/w)**2)) +  (b/(1 + ((t-t0+(dt + h))/w)**2)) + (c/(1 + ((t-t0-(dt + h))/w)**2))
    p2_dt = (a /(1 + ((t-t0)/w)**2)) + (b/(1 + ((t-t0+(dt - h))/w)**2)) + (c/(1 + ((t-t0-(dt - h))/w)**2))
    ddt = (p1_dt - p2_dt) / (2 * h) 

    grad[:,0] = da
    grad[:,1] = dw
    grad[:,2] = dt0
    grad[:,3] = db
    grad[:,4] = dc
    grad[:,5] = ddt
    return y, grad

d_pred_init, grad = calc_lorentz(p0, t)

p=p0.copy()
for j in range(20):
    pred,grad=calc_lorentz(p,t)
    r=d_true-pred
    err=(r**2).sum()
    r=np.matrix(r).T
    grad=np.matrix(grad)

    lhs=grad.T @ grad
    rhs=grad.T @ r
    dp=np.linalg.pinv(lhs) @ rhs
    for jj in range(p.size):
        p[jj]=p[jj]+dp[jj]

#Error estimate
noise = np.std(np.abs(pred - d_true))
N_1 = noise**(-2)*np.identity(len(p))
covariance_expectation_value = np.linalg.pinv( grad.T @ grad @ N_1 )
p_err = np.sqrt(np.diag(covariance_expectation_value))


print("a is " + str(p[0]) + " ± " + str(p_err[0]))
print("w is " + str(p[1]) + " ± " + str(p_err[1]))
print("t0 is " + str(p[2]) + " ± " + str(p_err[2]))
print("b is " + str(p[3]) + " ± " + str(p_err[3]))
print("c is " + str(p[4]) + " ± " + str(p_err[4]))
print("dt is " + str(p[5]) + " ± " + str(p_err[5]))

plt.figure(figsize=(16,9))
plt.plot(t,d_true, label="Our data")
plt.plot(t, d_pred_init, label="Initial step")
plt.plot(t, pred, label="Predicted data")
plt.xlabel("t")
plt.ylabel("d")
plt.legend()
plt.show()

plt.figure(figsize=(16,9))
plt.plot(t,d_true - pred)
plt.show()