import matplotlib.pyplot as plt
import math
import numpy as np

class Individual:
    def __init__(self, lifeTime, offspring):
        self.yearsLeft = lifeTime
        self.children = offspring
    def newYear(self):
        ans = []
        if self.yearsLeft > 0:
            ans = [Individual(3,1) for i in range(self.children)]
        self.yearsLeft -= 1
        return ans
    def getYears(self):
        return self.yearsLeft
population = [Individual(3,1)]
simLength = 20
years = 3
children = 1
time = 0
x = [0]
y = [1]
density = [y[0]/10]
while time < simLength:
    aliveIndiv = 0
    newIndiv = []
    for p in population:
        newIndiv.extend(p.newYear())
        if p.getYears() > 0:
            aliveIndiv = aliveIndiv + 1
    population.extend(newIndiv)
    y.append(len([p.yearsLeft for p in population if p.yearsLeft > 0]))
    time = time + 1
    x.append(time)
plt.plot(x,y)
plt.xlabel("Time")
plt.ylabel("Population size (Nt)")
plt.show()

density = [j/10 for j in y]
plt.plot(x,density)
plt.xlabel("Time")
plt.ylabel("Density (Nt/10m^2)")
plt.show()

space = [10/j for j in y]
plt.plot(x,space)
plt.xlabel("Time")
plt.ylabel("Space (10m^2/Nt)")
plt.show()

finiteRateIncrease = []
for i in range(1,len(y)):
    finiteRateIncrease.append(y[i]/y[i-1])
plt.xscale("log")
plt.plot(y[1:],finiteRateIncrease)
plt.scatter(y[1:],finiteRateIncrease)
plt.suptitle("Exponential Population Growth")
plt.xlabel("Population Size (Nt)")
plt.ylabel("Finite Rate of Increase")
plt.show()
print(finiteRateIncrease[7])
print(finiteRateIncrease[8])


#Plotting logistic growth finite rate increase
time = np.linspace(0,50,100)
c = 25
m = 210
popSize = [(math.exp(t-c)/ (1+math.exp(t-c))) * m for t in time]
finiteRateIncrease2 = [popSize[i]/popSize[i-1] for i in range (1,len(popSize))]
plt.suptitle("Logistic Population Growth")
plt.xlabel("Population Size (Nt)")
plt.ylabel("Finite Rate of Increase")
plt.plot(popSize[1:],finiteRateIncrease2)
plt.show()
print(finiteRateIncrease2[0])
