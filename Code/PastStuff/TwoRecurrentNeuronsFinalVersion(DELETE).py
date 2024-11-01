import time
import numpy as np
import matplotlib.pyplot as plt
import math

dt = 0.1 #Time step [ms]
t_stimStart = 100 #Time start box current [ms]
t_stimEnd = 500 #Time stop box current [ms]
t_end = 1000 #Time to stop the simulation [ms]

tau = 20 #membrane time constant [ms]
t_vect = np.arange(0, t_end, dt) #Creates time vector[ms] with time step dt[ms]

def Activation(a_param, p_param, r_param):
    numerator = r_param ** p_param
    denominator = a_param + r_param ** p_param
    return numerator/denominator
def Self(a_param, p_param, r_param):
    return r_param

#Running a simulation using a given nonlinear activation function on two recurrent neurons
#Constants for every run
weightMatrix = np.array([[0,0],[0,0]])
fixedA = 20
fixedP = 2
outputMatrix = np.zeros((2,len(t_vect)))
tIdx = 1
#Running the simulation for every weight in the weight values
pairs = np.empty()
while tIdx < len(t_vect):
    #Update rule for multiple connected nonlinear neurons    
    input_vect = outputMatrix[:,tIdx-1]
    outputMatrix[0][tIdx] = outputMatrix[0][tIdx-1] + dt/tau*(-outputMatrix[0][tIdx-1] + weightMatrix*Activation(fixedA,fixedP,input_vect)) #+ current1_vect[tIdx-1])
    outputMatrix[1][tIdx] = outputMatrix[1][tIdx-1] + dt/tau*(-outputMatrix[1][tIdx-1] + weightMatrix*Activation(fixedA,fixedP,input_vect)) #+ current1_vect[tIdx-1])
    pairs.append(input_vect)
    tIdx = tIdx + 1
plt.plot(pairs[:,0],pairs[:,1])
plt.show()
