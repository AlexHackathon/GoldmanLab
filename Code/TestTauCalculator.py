import numpy as np
import matplotlib.pyplot as plt
import  scipy as sp

def CalculateTau(x,y):
    #ATTENTION: five times the time constant produces a 3.5% underestimation error consistently,
    #Subtract min from all
    y = y - y[-1]
    #Scale so that the first value is one
    y = y/y[0]
    tau = sp.integrate.trapezoid(y,x)
    return tau

"""x = np.linspace(0,4000,2000)
tau = 800
y = np.zeros((len(x)))
for i in range(len(x)):
    y[i] = np.power(np.e, -x[i]/tau)
plt.plot(x,y)
plt.show()
#print(y)
print(CalculateTau(x,y))"""
