import numpy as np
import matplotlib.pyplot as plt

#Define Simulation Parameters
eyeMin = -20
eyeMax = 20
eyeRes = 4000
maxFreq = 80
totalTime = 1000 #14000 final length for tau calculation

dt = .01
P0=.1
t_f = 2000
t_s = 50
f = 1/(t_f*80)

tIdx=1
tVect = np.arange(0,totalTime,dt)
rVect = np.zeros(len(tVect))
sVect = np.zeros(len(tVect))
PRelVect = np.zeros(len(tVect))
def Inverse(sLast,rLast):
    return (P0 + t_f*f*rLast) / (1+t_f*f*rLast) / sLast
def FacFunc(rLast):
    return (P0 + t_f*f*rLast) / (1+t_f*f*rLast) * rLast

rVect[0] = 30
PRelVect[0] = P0
sVect[0] = FacFunc(rVect[0])

while tIdx < len(tVect):
    PRelVect[tIdx] = PRelVect[tIdx-1] + dt/t_f * (-PRelVect[tIdx-1] + P0 + t_f*f*rVect[tIdx-1]*(1-PRelVect[tIdx-1]))
    sVect[tIdx] = sVect[tIdx-1] + dt/t_s * (-sVect[tIdx-1] + PRelVect[tIdx-1]*rVect[tIdx-1])
    rVect[tIdx] = Inverse(rVect[tIdx-1],sVect[tIdx-1])
    if(tIdx < 10):
        print(rVect[tIdx],sVect[tIdx])
        input()
    tIdx = tIdx + 1

plt.plot(tVect, rVect)
plt.ylim((0,80))
plt.show()
plt.plot(tVect, sVect)
plt.ylim((0,80))
plt.show()