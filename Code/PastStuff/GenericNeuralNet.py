import numpy as np
import matplotlib.pyplot as plt
import math

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

class Neuron:
    def __init__(self, current_param=np.zeros(len(t_vect))):
        '''Sets the current injected over time and initalizes it's firing output vector'''
        self.current_vect = current_param
        self.v_vect = np.zeros(len(t_vect))
    def UpdateRule(self, weights_mat, tIdx_param, u_vect, dt_param, tau_param):
        '''Uses the linear update rule for multiple neurons'''
        totalInput = -self.v_vect[tIdx_param-1] + self.current_vect[tIdx_param-1] + sum(np.dot(u_vect, weights_mat))
        self.v_vect[tIdx_param] = self.v_vect[tIdx_param-1] + dt_param/tau_param*totalInput
    def UpdateRuleNonlinear(self, weights_mat, tIdx_param, u_vect, dt_param, tau_param, activationFunction):
        '''Uses a nonlinear update rule by passing the dot product of input and weights through an activation function'''
        totalInput = -self.v_vect[tIdx_param-1] + self.current_vect[tIdx_param-1] + activationFunction(sum(np.dot(u_vect, weights_mat)))
        self.v_vect[tIdx_param] = self.v_vect[tIdx_param-1] + dt_param/tau_param*totalInput
    def SigmoidActivation(self, current):
        return 1/(1+ exp(-current))
    def FractionActivation(self, current):
        return 1/(1-current)
        
tIdx = 1
#NONFUNCTIONAL
neuronPop = [] #Array of neurons in the same order as they appear in the weights
#NONFUNCTIONAL
weightMatrix = np.array([[0,1],
                        [1,0]]) #Matrix of weights based on the neuron population {len(neuronPop)xlen(neuronPop)}
#Iterates through time
#Gets the output of all neurons at the previous time frame
#Goes through each neuron at a time frame
#Calculates each neuron's firing rate based on the input
while tIdx < len(t_vect):
    u_vect_temp = np.array([])
    for n in neuronPop:
        u_vect_temp.append(n.v_vect[tIdx-1])
    for n in neuronPop:
        n.UpdateRule(weightMatrix, tIdx, u_vect_temp, dt, tau)
    tIdx = tIdx + 1
