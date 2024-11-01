import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression

dt = 0.1 #Time step [ms]
t_stimStart = 100 #Time start box current [ms]
t_stimEnd = 500 #Time stop box current [ms]
t_end = 1000 #Time to stop the simulation [ms]
I_e = 1.43 #Magnitude of current

tau = 20 #membrane time constant [ms]
t_vect = np.arange(0, t_end, dt) #Creates time vector[ms] with time step dt[ms]

#Creates a box current over the time vector with magnitude stimMag
def ConstCurrent(time_vect, stimMag, stimStartEnd_vect):
    addCurrent = False
    i = 0
    current_vect = np.zeros(len(time_vect))
    for x in range(0, len(time_vect)-1):
        if i >= len(stimStartEnd_vect):
            continue
        elif time_vect[x] >= stimStartEnd_vect[i]:
            addCurrent = not addCurrent
            i = i + 1
        if addCurrent:
            current_vect[x] = stimMag
    return current_vect

def Activation(a_param, p_param, r_param):
    numerator = r_param ** p_param
    denominator = a_param + r_param ** p_param
    return numerator/denominator
def Self(a_param, p_param, r_param):
    return r_param

#Begin linear regression fitting
#Global simulation variables that show what to calculate
plotCurrent = False #Should the current be plotted (NOT IMPLEMENTED)
plotFiringRate = False #Should the nonlinear firing rate with different weights be plotted
plotTeff = False #Should t_eff vs weight be plotted
plotNonlinear = True #Should the nonlinear functions and how they vary be plotted

#Variables for linear regression
timeCalcStart = 550 #Time to start fitting from [ms]
trainTIdx = np.searchsorted(t_vect, timeCalcStart) #Index where time is greater than timeCalcStart ms
deltaT = t_vect[trainTIdx:] - t_vect[trainTIdx] #Array of delta ts from t0=timeCalcStart
deltaT = deltaT.reshape(-1,1)

#Constants for every run
current1_vect = ConstCurrent(t_vect, 30, [t_stimStart, t_stimEnd]) #Current injected into the neuron [spikes/sec]
weightValues = np.arange(-1, .99, 0.01)
t_eff = np.zeros(len(weightValues))
tauIdx = 0

#Running the simulation for every weight in the weight values
for w in weightValues:
    if not plotTeff:
        break
    v1a_vect = np.zeros(len(t_vect))
    tIdx = 1
    #Update rule for autapse
    while tIdx < len(t_vect):
        totalInput = v1a_vect[tIdx-1]*w + current1_vect[tIdx-1]
        v1a_vect[tIdx] = v1a_vect[tIdx-1] + dt/tau*(-v1a_vect[tIdx-1] + totalInput)
        tIdx = tIdx + 1
    #Set the x and y variables for regression including reshaping
    yReg = np.log(v1a_vect[trainTIdx:])
    yReg = yReg.reshape(-1,1)
    #Fit the linear regression
    reg = LinearRegression()
    reg.fit(deltaT,yReg)
    t_eff[tauIdx] = -1/reg.coef_[0][0]
    tauIdx = tauIdx + 1
if plotTeff:
    plt.plot(weightValues, t_eff)
    plt.suptitle("Relationship Between Tau Effective and Synaptic Weights")
    plt.xlabel("Recurrent Synaptic Weight")
    plt.ylabel("Tau Effective")
    plt.show()
#Running a simulation for different values of nonlinear activation functions
r = np.linspace(0,1, 100)
a_vals = np.arange(1,50, 2)
p_vals = np.arange(1,4, 1)
y_a = np.zeros(len(r))

if plotNonlinear:
    fig, axs = plt.subplots(ncols=2)
    fig.suptitle("Activation Function r^p/(a+r^p)")

    for a in a_vals:
        for i in range(0,len(r)):
            y_a[i] = r[i]/(a+r[i])
        axs[0].plot(r, y_a, label=str(a))
    axs[0].set_title("Varrying a Values With p=1")
    axs[0].legend()
    axs[0].plot()

    for p in p_vals:
        for i in range(0,len(r)):
            y_a[i] = r[i]**p/(20+r[i]**p)
        axs[1].plot(r, y_a, label=str(p))
    axs[1].set_title("Varrying p Values With a=20")
    axs[1].legend()
    axs[1].plot()
    plt.show()

#Running a simulation using a given nonlinear activation function
#Constants for every run
weightValuesNonlinear = [-1, -.5, 0, .5]

if plotFiringRate:
    #Running the simulation for every weight in the weight values
    fig, axs = plt.subplots(1)
    fig.suptitle("Firing Rate Over Time With Varying Synaptic Weights")
    plotIdx = 0
    a = 40
    p = 1
    for w in weightValuesNonlinear:
        v2a_vect = np.zeros(len(t_vect))
        tIdx = 1
        #Update rule for autapse
        while tIdx < len(t_vect):
            totalInput = v2a_vect[tIdx-1]*w + current1_vect[tIdx-1]
            if tIdx > 4500 and tIdx < 4510:
                print(totalInput)
            v2a_vect[tIdx] = v2a_vect[tIdx-1] + dt/tau*(-v2a_vect[tIdx-1] + Activation(a,p,totalInput))
            tIdx = tIdx + 1
        axs.plot(t_vect, v2a_vect, label=str(w))
        axs.set_ylabel("Firing rate")
        axs.set_xlabel("Time")
    axs.legend()
    plt.show()

#Running a simulation for finding fixed points
rPotentialValues = np.linspace(0,1,50)
externalInput = 30
wVals = [-1,-.5,0,.5,1]
for w in wVals:
    activationTransformation = [Activation(.3,1,r*w + externalInput) for r in rPotentialValues]
    y = (-rPotentialValues + activationTransformation) / tau
    plt.plot(rPotentialValues, y, label=str(w))
plt.legend()
plt.show()
