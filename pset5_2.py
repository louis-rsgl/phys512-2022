import numpy as np
from matplotlib import pyplot as plt
import camb

newtom_params_txt_file = "planck_fit_params.txt"

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
pars=np.asarray([69, 0.022, 0.12,0.06, 2.1e-9, 0.95])
planck=np.loadtxt('Assignment5/COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0] #x
spec=planck[:,1] #y

dx = 1e-8
dp = pars*dx
fitp,curve=newton(get_spectrum,pars,dp,ell,spec)

error_fitp=np.diag(np.linalg.cholesky(curve))
errs=0.5*(planck[:,2]+planck[:,3])
model=get_spectrum(fitp)
model=model[:len(spec)]
resid=spec-model
chisq=np.sum((resid/errs)**2)
print("chsiq is " + str(chisq))

f = open("Assignment5/planck_fit_params.txt", "a")
line = "Hubble constant is " + str(fitp[0]) + " ± " + str(error_fitp[0]) + "\n"
f.write(line)
line = "Baryon density is " + str(fitp[1]) + " ± " + str(error_fitp[1]) +"\n"
f.write(line)
line = "Dark matter density is " + str(fitp[2]) + " ± " + str(error_fitp[2]) +"\n"
f.write(line)
line = "Optical depth is " + str(fitp[3]) + " ± " + str(error_fitp[2])+ "\n"
f.write(line)
line = "Primordial amplitude of the spectrum is " + str(fitp[4]) + " ± " + str(error_fitp[4])+ "\n"
f.write(line)
line = "Primordial tilt of the spectrum is " + str(fitp[5]) + " ± " + str(error_fitp[5])
f.write(line)