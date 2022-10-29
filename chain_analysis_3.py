import numpy as np
import matplotlib.pyplot as plt

f = open("Assignment5/chain_3.txt", "r")
chain = []
lines = f.readlines()
d = open("Assignment5/chain_chi_3.txt", "r")
lines_d = d.readlines()
chain_chi = []
for i in range(len(lines_d)):
    chain_chi.append(float(lines_d[i]))

count_chain = 0
for r in range(0,len(lines),2):
    line1 = lines[r][1:].split(" ")
    line2 = lines[r+1][1:-2].split(" ")
    line1.append(line2[0])
    line1.append(line2[1])
    line = []
    line.append(chain_chi[count_chain])
    count_chain = count_chain + 1
    for j in line1:
        line = np.append(line,float(j))
    chain.append(np.array(line))
chain = np.array(chain)
steps = np.linspace(0,20000, 20000)

#for i in range(len(chain.T)):
    #plt.plot(steps, chain.T[i])
re = open("Assignment5/planck chain.txt", "a")

for i in range(len(chain)):
    line = ""
    for j in range(len(chain[i])):
        line = line + " " + str(chain[i][j]) + " "
    line = line + str("\n")
    re.write(line)

