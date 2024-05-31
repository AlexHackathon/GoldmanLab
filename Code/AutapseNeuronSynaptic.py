import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.animation import FuncAnimation
import plotly.express as px
import plotly.graph_objects as go
import math
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
def ActivationE(r_input, r_m=0, width_param=1):
    #Default is the sigmoid with no change in width or center
    num = np.exp((r_input - r_m)/width_param)
    denom= 1 + num
    return num/denom  
class Simulation:
    def __init__(self, synapseNum, dt, stimStart, stimEnd, end, tau, I_e, T):
        '''Instantiates the simulation
           ---------------------------
           Paramters
           neuronNum: the number of neurons simulated
           dt: the time step of the simulation in ms
           t_stimStart: the start of the electric current stimulation
           t_stimEnd: the end of the electric current stimulation
           t_end: the end of the simulation
           tau: the neuron time constant for all neurons
           currentMatrix: a matrix with each row representing the injected current of a neuron over time'''
        self.synapseNum = synapseNum #Number of neurons in the simulation
        self.dt = dt #Time step [ms]
        self.t_stimStart = stimStart #Stimulation start [ms]
        self.t_stimEnd = stimEnd #Stimulation end [ms]
        self.t_end = end #Simulation end [ms]
        self.tau = tau
        self.t_vect = np.arange(0, self.t_end, self.dt)
        self.current_vect = ConstCurrent(self.t_vect, I_e, [self.t_stimStart, self.t_stimEnd])
        self.w_vect = np.zeros(self.synapseNum)
        self.v_vect = np.zeros(len(self.t_vect))
        self.tonic = T
    def SetWeightVectRand(self, scaling, shift, seed=0):
        #np.random.seed(seed)
        #for i in range(len(self.w_vect)):
        #    print(np.random.randn())
        #    print(np.random.randn()*scaling)
        #    print(np.random.randn()*scaling + shift)
        #    self.w_vect[i]=np.random.randn()*scaling + shift
        #print(self.w_vect)
        self.w_vect = np.array([20,20,20,20,20])
    def RunSim(self):
        tIdx = 1
        while tIdx < len(self.t_vect):
            v = self.v_vect[tIdx-1]
            i = self.current_vect[tIdx-1]
            S_r = np.array([ActivationE(v, r_m=10, width_param=5),
                            ActivationE(v, r_m=30, width_param=5),
                            ActivationE(v, r_m=50, width_param=5),
                            ActivationE(v, r_m=70, width_param=5),
                            ActivationE(v, r_m=90, width_param=5)])
            v_t = v + self.dt/self.tau*(-v + np.dot(S_r, self.w_vect) + i + self.tonic)
            self.v_vect[tIdx] = v_t
            tIdx = tIdx + 1
    def PlotSim(self):
        plt.plot(self.t_vect, self.v_vect)
        plt.show()
    def PlotNullcline(self, I):
        v = np.linspace(0,100,200)
        dot_val = []
        for val in v:
            #S_r = np.array([ActivationE(val, r_m = 50, width_param=1) for i in range(self.synapseNum)])
            S_r = np.array([ActivationE(val, r_m=10, width_param=5),
                            ActivationE(val, r_m=30, width_param=5),
                            ActivationE(val, r_m=50, width_param=5),
                            ActivationE(val, r_m=70, width_param=5),
                            ActivationE(val, r_m=90, width_param=5)])
            dot_val.append(np.dot(S_r, self.w_vect))
            '''if(val==0):
                print(ActivationE(val, r_m = 50, width_param=1))
                print(np.dot(S_r, self.w_vect))'''
        dot_val = np.array(dot_val)
        nullcline = self.dt/self.tau*(-v + dot_val + I + self.tonic)
        #print(nullcline[0])
        plt.plot(v,dot_val, label="F(r) = sum(w_i * S(r))")
        plt.plot(v,v, label="F(r) = r")
        plt.suptitle("Finding Fixed Points for a Nonlinear Synaptic Autapse")
        plt.legend()
        plt.xlabel("Firing Rate (spikes/sec)")
        plt.show()
        plt.suptitle("Finding Fixed Points for a Nonlinear Synaptic Autapse")
        plt.plot(v, nullcline)
        plt.plot(v, np.zeros(len(v)))
        plt.xlabel("Firing Rate (spikes/sec)")
        plt.ylabel("dt/tau*(-r+sum(w_i * S(r))+I+T")
        plt.show()
#****************************************************************************
#Actual simulation code
#Order: synapseNum, dt, stimStart, stimEnd, end, tau, currentMag, T(constant tonic input)
sim = Simulation(5, .1, 100, 600, 3000, 20, 2, 0)
sim.SetWeightVectRand(20, 0)
sim.RunSim()
sim.PlotSim()
sim.PlotNullcline(.2)
