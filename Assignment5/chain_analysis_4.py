#Code to format correctly the pars data
##it also compares the chain we had in 3 to the chain 4
import numpy as np
import matplotlib.pyplot as plt

f1 = open("Assignment5/OUTPUT_FIXED_CHAIN_4/fixed_chain_4.txt", "r")
f2 = open("Assignment5/OUTPUT_CHAIN_3/chain_3.txt", "r")
chain1 = []
lines_f1 = f1.readlines()
chain2 = []
lines_f2 = f2.readlines()
d1 = open("Assignment5/OUTPUT_FIXED_CHAIN_4/fixed_chain_chi_4.txt", "r")
lines_d1 = d1.readlines()
chain_chi1 = []
d2 = open("Assignment5/OUTPUT_CHAIN_3/chain_chi_3.txt", "r")
lines_d2 = d2.readlines()
chain_chi2 = []

for i in range(len(lines_d1)):
    chain_chi1.append(float(lines_d1[i]))
for i in range(len(lines_d2)):
    chain_chi2.append(float(lines_d2[i]))
    
def chain_creat(lines_f1,chain1):
    count_chain = 0
    for r in range(0,len(lines_f1),2):
        line1 = lines_f1[r][1:].split(" ")
        line2 = lines_f1[r+1][1:-2].split(" ")
        line1.append(line2[0])
        line1.append(line2[1])
        line = []
        count_chain = count_chain + 1
        for j in line1:
            line = np.append(line,float(j))
        chain1.append(np.array(line))
    chain1 = np.array(chain1)
    return chain1
chain1 = chain_creat(lines_f1,chain1)
chain2 = chain_creat(lines_f2,chain2)

#Text editting
re = open("Assignment5/planck_chain_tauprior.txt", "a")

for i in range(len(chain1)):
    line = ""
    for j in range(len(chain1[i])):
        line = line + " " + str(chain1[i][j]) + " "
    line = line + str("\n")
    re.write(line)


#Plot chi question 4 v steps
steps = np.linspace(0,20000, 20000)
plt.plot(steps, chain_chi1)

#Plot chain[i] question 4 v steps
for i in range(len(chain1.T)):
    plt.plot(steps, chain1.T[i])
plt.show()

#Comparison
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
mean,errs,wts=process_chain(chain1,chain_chi1)
mean2,errs2,wts2=process_chain(chain2,chain_chi2)
nsig=5
npar=chain1.shape[1]
for i in range(npar):
    t1=mean[i]+errs[i]*nsig
    t2=mean[i]-errs[i]*nsig
    frac=(np.sum(chain1[:,i]>t1)+np.sum(chain1[:,i]<t2))/chain1.shape[0]
    frac2=(np.sum(chain2[:,i]>t1)+np.sum(chain2[:,i]<t2))/chain2.shape[0]
    print('fractions of samples on param ',i,' more than ',nsig,' is ',frac,frac2)

