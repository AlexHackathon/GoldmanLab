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

'''class Neuron:
    def __init__(self, tau_param, u_param):
        self.tau = tau_param 
        self.v_vect = np.zeros(len(t_vect)) #Creates firing output vector
        self.v_vect[0] = 0 #Initial firing rate is 0 spikes/sec
        self.u_vect = u_param #Sets the input vector
        self.w_vect = inputWeightsVect_param #Sets weight vector
    def PlotSelf(self, currentInjection_param, firingRate_param):
        #Plots the current with respect to time
        currentInjection = ConstCurrent(t_vect, I_e, [t_stimStart,t_stimEnd])
        fig, axs = plt.subplots(2, constrained_layout = True)
        fig.suptitle('Box Current and Corresponding Firing Rate')
        axs[0].plot(t_vect, currentInjection)
        axs[0].set_title("Current x Time")
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Current [nA]")

        #Plots firing rate over time using step-wise updating
        for i in range(1, len(t_vect)-1):
            firingRate_vect[i] = dt * (-firingRate_vect[i-1] + currentInjection[i-1]) * 1/tau + firingRate_vect[i-1]
        axs[1].plot(t_vect, firingRate_vect)
        axs[1].set_title("Firing Rate x Time")
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Firing rate [spikes/second]")
        plt.show()
    def InjectionCalculation(self):
        #Calculates the firing rate over entire t_vect after an injection modelled by an input current
    def SynapticCalculation(self, u_param, w_param):
        #Calculates the firing rate based on synaptic input
    def SynapticCurrentConstruction(self):
        #Take the inputs and calculate all the synaptic current over time
    def RateConstruction(self):
        #Take the synaptic current and use the update rule and a sigmoid to update the firing rate
    def GetV(self):
        return self.v_vect'''

#Code Plan
#Store the neurons in an multidimensional array with the first level being 0
#For each level, determine firing rate at time t=dt (initally everything is 0)
#Move up a level and use inputs to calculate the firing rates using the u
'''
neuronalStructure = [[Neuron()],[Neuron()]]
def InputConstruction(levelIdx_param, neuronIdx_param):
    #Constructs the u vector based on all neurons that feed into it.
       Relies on the neuronalStructure to extract which neurons have dependencies on which other neurons
#Iterate through the array to set the synpatic current
for t in t_vect:
    for levelIdx in range(len(neuronalStructure)-1):
        for nIdx in range(len(levelIdx)-1):
            u_vect = neuron.SynapticCalculation(InputConstruction(levelIdx, nIdx))
            #Insert stepwise updating for the 
#Iterate through a second time to use the synaptic current to update the  
for t in t_vect:
    for levelIdx in range(len(neuronalStructure)-1):
        for nIdx in range(len(levelIdx)-1):
            u_vect = neuron.SynapticCalculation(InputConstruction(levelIdx, nIdx))
            #Insert stepwise updating for the'''


#**********************************************************************************************************************\
def ActivationFunctionSigmoid(I_s_param):
    return 1/(1+math.exp(-I_s_param))
def ActivationFunctionSelf(I_s_param):
    return I_s_param
def UpdateRuleSynapticSigmoid(u_vect_param, w_vect_param, v_prev, dt_param, tau_param):
    return v_prev + dt_param/tau_param*(-v_prev + ActivationFunctionSigmoid(np.dot(u_vect_param, w_vect_param)))
def UpdateRuleSynapticSelf(u_vect_param, w_vect_param, v_prev, dt_param, tau_param):
    return v_prev + dt_param/tau_param*(-v_prev + ActivationFunctionSelf(np.dot(u_vect_param, w_vect_param)))

'''#Neuron 1
u1_vect = ConstCurrent(t_vect, 30, [t_stimStart, t_stimEnd])
w1 = 1
v1_vect = np.zeros(len(t_vect))
tIdx = 1
while tIdx < len(t_vect):
    v1_vect[tIdx] = UpdateRuleSynaptic(u1_vect[tIdx-1], w1, v1_vect[tIdx-1], dt, tau)
    tIdx = tIdx + 1

plt.plot(t_vect, v1_vect)
plt.plot(t_stimStart + tau, .63 * 30, 'ro')
plt.show()

#Neuron 2
w2 = 2
v2_vect = np.zeros(len(t_vect))
tIdx = 1
while tIdx < len(t_vect):
    v2_vect[tIdx] = UpdateRuleSynaptic(v1_vect[tIdx-1], w2, v2_vect[tIdx-1], dt, tau)
    tIdx = tIdx + 1
plt.plot(t_vect, v2_vect)
#plt.plot(t_stimStart + tau, .63 * 30, 'ro')
plt.show()'''

#*************************************************************************************************************************
'''#Neuron 1a
u1a_vect = ConstCurrent(t_vect, 30, [t_stimStart, t_stimEnd])
w1a = 1
v1a_vect = np.zeros(len(t_vect))
tIdx = 1
while tIdx < len(t_vect):
    v1a_vect[tIdx] = UpdateRuleSynapticSelf(u1a_vect[tIdx-1], w1a, v1a_vect[tIdx-1], dt, tau)
    tIdx = tIdx + 1

plt.plot(t_vect, v1a_vect)
plt.plot(t_stimStart + tau, .63 * 30, 'ro')
plt.show()

#Neuron 1b
u1b_vect = ConstCurrent(t_vect, 30, [t_stimStart, t_stimEnd])
w1b = 1
v1b_vect = np.zeros(len(t_vect))
tIdx = 1
while tIdx < len(t_vect):
    v1b_vect[tIdx] = UpdateRuleSynapticSelf(u1b_vect[tIdx-1], w1b, v1b_vect[tIdx-1], dt, tau)
    tIdx = tIdx + 1

plt.plot(t_vect, v1b_vect)
plt.plot(t_stimStart + tau, .63 * 30, 'ro')
plt.show()

#Neuron 2
w2 = [1,0.5]
v2_vect = np.full(len(t_vect),0.5)
tIdx = 1
while tIdx < len(t_vect):
    myU_vect = [v1a_vect[tIdx-1],v1a_vect[tIdx-1]]
    v2_vect[tIdx] = UpdateRuleSynapticSelf(myU_vect, w2, v2_vect[tIdx-1], dt, tau)
    tIdx = tIdx + 1
plt.plot(t_vect, v2_vect)
#plt.plot(t_stimStart + tau, .63 * 30, 'ro')
plt.show()'''
#***********************************
#Neuron 1
u1a_vect = ConstCurrent(t_vect, 30, [t_stimStart, t_stimEnd])
w1a = 1
v1a_vect = np.zeros(len(t_vect))
tIdx = 1
while tIdx < len(t_vect):
    v1a_vect[tIdx] = UpdateRuleSynapticSelf(u1a_vect[tIdx-1], w1a, v1a_vect[tIdx-1], dt, tau)
    tIdx = tIdx + 1

w2a = 2
tIdx = 1
while tIdx < len(t_vect):
    v1a_vect[tIdx] = UpdateRuleSynapticSelf(u1a_vect[tIdx-1], w2a, v1a_vect[tIdx-1], dt, tau)
    tIdx = tIdx + 1

plt.plot(t_vect, v1a_vect)
plt.plot(t_stimStart + tau, .63 * 30, 'ro')
plt.show()


                                                          
                                                                                                    
