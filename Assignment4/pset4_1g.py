import numpy as np
from matplotlib import pyplot as plt

scale= [2.73234500e-04, 4.87277167e-09, 3.44357572e-09]
data=np.load('Assignment4/sidebands.npz')
def get_step(trial_step):
    if len(trial_step.shape)==1:
        return np.random.randn(len(trial_step))*trial_step
    else:
        L=np.linalg.cholesky(trial_step)
        return L@np.random.randn(trial_step.shape[0])


def gauss(pars,t):
    amp=pars[0]
    t0=pars[1]
    w=pars[2]
    y = amp/(1 + ((t-t0)/w)**2)
    return y

def gauss_chisq(pars,data):
    
    t=np.array(data['time'])
    d_true=np.array(data['signal'])
    pred=gauss(pars,t)
    errs=np.std(np.abs(pred-d_true))
    chisq=np.sum((pred-d_true)**2/errs**2)
    return chisq

def num_derivs(fun,pars,dp,x):
    A=np.empty([len(x),len(pars)])
    for i in range(len(pars)):
        pp=pars.copy()
        pp[i]=pars[i]+dp[i]
        y_right=fun(pp,x)
        pp[i]=pars[i]-dp[i]
        y_left=fun(pp,x)
        A[:,i]=(y_right-y_left)/(2*dp[i])
    return A

def newton(fun,pars,dp,x,y,niter=10):
    for i in range(niter):
        pred=fun(pars,x)
        r=y-pred
        A=num_derivs(fun,pars,dp,x)
        lhs=A.T@A
        rhs=A.T@r
        step=np.linalg.inv(lhs)@rhs
        pars=pars+step
        print('step is ',step)
    return pars,np.linalg.inv(lhs)

def run_chain(fun,pars,trial_step,data,nstep=20000,T=1):
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chisq=np.zeros(nstep)
    chain[0,:]=pars
    chi_cur=fun(pars,data)
    chisq[0]=chi_cur
    for i in range(1,nstep):
        pp=pars+get_step(trial_step)
        new_chisq=fun(pp,data)
        accept_prob=np.exp(-0.5*(new_chisq-chi_cur)/T)
        if np.random.rand(1)<accept_prob:
            pars=pp
            chi_cur=new_chisq
        chain[i,:]=pars
        chisq[i]=chi_cur
    return chain,chisq


def process_chain(chain,chisq,T=1.0):
    dchi=chisq-np.min(chisq)
    #density in chain is exp(-0.5*chi^2/T), but
    #we wanted it to be exp(-0.5*chi^2)
    #so, we want to downweight by ratio, which is
    #exp(-0.5*chi^2*(1-1/T)).  We'll calculate the mean
    #and standard deviation of the chain, but will also
    #return the weights so you could calculate whatever you want

    wt=np.exp(-0.5*dchi*(1-1/T)) #the magic line that importance samples

    #calculate the weighted sum of the chain and the chain squared
    npar=chain.shape[1]
    tot=np.zeros(npar)
    totsqr=np.zeros(npar)
    for i in range(npar):
        tot[i]=np.sum(wt*chain[:,i])
        totsqr[i]=np.sum(wt*chain[:,i]**2)
    #divide by sum or weights
    mean=tot/np.sum(wt)
    meansqr=totsqr/np.sum(wt)

    #variance is <x^2>-<x>^2
    var=meansqr-mean**2
    return mean,np.sqrt(var),wt
data=np.load('Assignment4/sidebands.npz')

t=np.array(data['time'])
d_true=np.array(data['signal'])

pars=np.asarray([1.422,0.000192,1.792e-05])
dp=np.asarray([0.01,0.0001,0.00001])
fitp,curve=newton(gauss,pars,dp,t,d_true)

chain,chivec=run_chain(gauss_chisq,fitp,curve,data)
mean,errs,wts=process_chain(chain,chivec)
print(errs)
steps = np.linspace(0,20000, 20000)
plt.plot(steps, chivec)
plt.show()