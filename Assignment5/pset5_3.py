#Code to compute MCMC

import numpy as np
from matplotlib import pyplot as plt
import camb
import corner

def get_spectrum(pars,lmax=3000):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    
    return tt[2:]
def num_derivs(fun,pars,dp,x):
    A=np.empty([len(x),len(pars)])
    for i in range(len(pars)):
        pp=pars.copy()
        pp[i]=pars[i]+dp[i]
        y_right=fun(pp)
        y_right=y_right[:len(spec)]
        pp[i]=pars[i]-dp[i]
        y_left=fun(pp)
        y_left=y_left[:len(spec)]
        A[:,i]=(y_right-y_left)/(2*dp[i])
    return A
def newton(fun,pars,dp,x,y,niter=3):
    errs=0.5*(planck[:,2]+planck[:,3])
    err_diag = np.asarray(errs**2)
    N = np.zeros((len(x),len(x)))
    np.fill_diagonal(N, err_diag)
    for i in range(niter):
        pred=fun(pars)
        pred=pred[:len(spec)]
        r=y-pred
        A=num_derivs(fun,pars,dp,x)
        lhs=A.T@np.linalg.inv(N)@A
        rhs=A.T@np.linalg.inv(N)@r
        step=np.linalg.inv(lhs)@rhs
        pars=pars+step
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
def get_step(trial_step):
    if len(trial_step.shape)==1:
        return np.random.randn(len(trial_step))*trial_step
    else:
        L=np.linalg.cholesky(trial_step)
        return L@np.random.randn(trial_step.shape[0])
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
def spectrum_chisq(pars,data):
    x=data['x']
    y=data['y']
    errs=data['errs']
    
    model=get_spectrum(pars)
    model=model[:len(spec)]
    resid=spec-model
    chisq=np.sum((resid/errs)**2)
    return chisq
  
#Initial pars
pars=np.asarray([69, 0.022, 0.12,0.06, 2.1e-9, 0.95])

#DATA
planck=np.loadtxt('Assignment5/COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0] #x
spec=planck[:,1] #y

#Setup
dx = 1e-8
dp = pars*dx
fitp,curve=newton(get_spectrum,pars,dp,ell,spec)

#Errors on pars
error_fitp=np.diag(np.linalg.cholesky(curve))
errs=0.5*(planck[:,2]+planck[:,3])
model=get_spectrum(fitp)
model=model[:len(spec)]
resid=spec-model
chisq=np.sum((resid/errs)**2)

data={}
data['x']=ell
data['y']=spec
data['errs']=errs

#Run MCMC chain
chain,chivec=run_chain(spectrum_chisq,fitp,curve,data)
steps = np.linspace(0,20000, 20000)
f = open("Assignment5/chain_4.txt", "a")
for line in chain:
    f.write(str(line) + "\n")
f.close()
f = open("Assignment5/chain_chi_4.txt", "a")
for line in chivec:
    f.write(str(line)+ "\n")
f.close()
f = open("Assignment5/pars_chain_4.txt", "a")
f.write(str(pars))
f.close()
f = open("Assignment5/errs_chain_4.txt", "a")
f.write(str(errs))
f.close()

#Chi v step graph showing the chain
plt.plot(steps, chivec)

#Corner of the distribution
corner.corner(chain)

#Log(FFT) plot
for i in range (len(chain.T)):
    plt.loglog(np.linspace(0,1,len(chain.T[i])),np.abs(np.fft.fft(chain.T[i]))**2)
plt.show()
