import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.animation import FuncAnimation
import plotly.express as px
import plotly.graph_objects as go
import math
import MyEig as myEig
#Creates a box current over the time vector with magnitude stimMag
def ConstCurrent(time_vect, stimMag, stimStartEnd_vect):
    addCurrent = False
    i = 0
    current_vect = np.zeros(len(time_vect))
    for x in range(0, len(current_vect)): #len had -1 don't know why
        if i >= len(stimStartEnd_vect):
            continue
        elif time_vect[x] >= stimStartEnd_vect[i]:
            addCurrent = not addCurrent
            i = i + 1
        if addCurrent:
            current_vect[x] = stimMag
    return current_vect
def ConstCurrentMat(time_vect, eigenDataParam, stimStartEnd_vect):
    finalCurr = []
    for s in eigenDataParam.GetVect():
        finalCurr.append(ConstCurrent(time_vect, s * 6, stimStartEnd_vect))
    return np.array(finalCurr)
class Simulation:
    def __init__(self, neuronNum, dt, stimStart, stimEnd, end, tau):
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
        self.neuronNum = neuronNum #Number of neurons in the simulation
        self.dt = dt #Time step [ms]
        self.t_stimStart = stimStart #Stimulation start [ms]
        self.t_stimEnd = stimEnd #Stimulation end [ms]
        self.t_end = end #Simulation end [ms]
        self.tau = tau
        self.t_vect = np.arange(0, self.t_end, self.dt)
        self.current_mat = None
        self.w_mat = np.zeros((self.neuronNum, self.neuronNum))
        self.v_mat = np.zeros((len(self.t_vect), neuronNum))
        self.eig=[]
    def SetCurrent(self, I_e):
        print(np.shape(I_e))
        self.current_mat = I_e
    def TakeNeuron(self):
        x1Int = True
        while x1Int:
            x1 = input("Neuron: ")
            if x1=="break":
                break
            try:
                x1 = int(x1)
                x1Int = False
            except:
                print("Not a valid neuron")
            if x1 >= neuronNum:
                print("Not a valid neuron")
                x1Int = True
        return x1
    def SetWeightMatrixRand(self, shift, scaling, seed):
        np.random.seed(seed)
        for i in range(len(self.w_mat)):
            for j in range(len(self.w_mat[i])):
                self.w_mat[i][j]=(np.random.randn()+ shift) * scaling
        evalue, evect = np.linalg.eig(self.w_mat)
        for i in range(len(evalue)):
            self.eig.append(myEig.EigenData(evalue[i], evect[:,i]))
    def SetWeightMatrixManual(self, wMatParam):
        self.w_mat = wMatParam
    def RunSim(self, v0 = [0,0]):
        tIdx = 1
        self.v_mat[0] = v0
        while tIdx < len(self.t_vect):
            v = self.v_mat[tIdx-1]
            i = self.current_mat[:,tIdx-1]
            v_vect_t = v + self.dt/self.tau*(-v + np.dot(v, self.w_mat) + i)
            self.v_mat[tIdx] = v_vect_t
            tIdx = tIdx + 1
        
    def GraphEig(self, lines=True):
        #fig = go.Figure()
        #layout = go.Layout(title=go.layout.Title(text="Eigenvalue Plot for 100 Neuron Network"))
        #fig.update_layout(xaxis_title="Real", yaxis_title="Imaginary")
        #for val in self.evalue:
        #    fig.add_trace(go.Scatter(x=[0,val.real],y=[0,val.imag]))
        #fig.show()
        if lines:
            for val in self.eig:
                plt.plot([0,val.GetValue().real],[0,val.GetValue().imag])
                if round(val.GetValue().real,5) == 1:
                    plt.annotate(xy =(val.GetValue().real,val.GetValue().imag), text=str(round((val.GetValue().real**2 + val.GetValue().imag**2)**(1/2),12)))
                else:
                    plt.annotate(xy =(val.GetValue().real,val.GetValue().imag), text=str(round((val.GetValue().real**2 + val.GetValue().imag**2)**(1/2),2)))
        else:
            for val in self.eig:
                plt.scatter([0,val.GetValue().real],[0,val.GetValue().imag])
        plt.suptitle("Eigenvalue Plot for 100 Neuron Network")
        #plt.xlim(-1.1,1.1)
        #plt.ylim(-1.1,1.1)
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.show()
        
    def GraphNeuronsTime(self):
        while True:
            x1 = input("Number of neurons to graph: ")
            if x1=="break":
                break #Doesn't work
            try:
                x1 = int(x1)
            except:
                print("Not a valid neuron")
                continue
            for i in range(x1):
                nIdx=self.TakeNeuron()
                if nIdx == "break":
                    break
                plt.plot(self.t_vect, self.v_mat[:,nIdx], label="Neuron "+str(nIdx))
            plt.xlabel("Time (ms)")
            plt.ylabel("Firing rate (spikes/sec)")
            plt.legend()
            plt.suptitle("Firing Rate Over Time")
            plt.show()
    def GraphNeuronsTogether(self):
        while True:
            nIdx1=self.TakeNeuron()
            if nIdx1 == "break":
                break
            nIdx2=self.TakeNeuron()
            if nIdx2 == "break":
                break
            plt.plot(self.v_mat[:,nIdx1], self.v_mat[:,nIdx2])
            plt.xlabel("Neuron "+str(nIdx1))
            plt.ylabel("Neuron "+str(nIdx2))
            plt.suptitle("Neuron "+str(nIdx1) + " x " + "Neuron "+str(nIdx2))
            plt.show()
    def PrintVect(self, key):
        print(self.eig[key])
#****************************************************************************
#Actual simulation code
'''neuronNum = 100
#Order: neuronNum, dt, stimStart, stimEnd, end, tau
sim = Simulation(neuronNum, .1, 100, 500, 1000, 20)
sim.SetWeightMatrixRand(1, 1/99.0029, 69)
integratorModes = []
for e in sim.eig:
    if e.IsRealOne(1):
        print(e.GetValue())
    if e.IsRealOne(3):
        integratorModes.append(e)
sim.GraphEig(lines=False)
print(len(integratorModes))
if len(integratorModes) == 0:
    quit()
x = ConstCurrentMat(sim.t_vect, integratorModes[0], [sim.t_stimStart, sim.t_stimEnd])
sim.SetCurrent(x)
sim.RunSim()
sim.GraphNeuronsTime()
sim.GraphNeuronsTogether()'''
#******************************************************************************
#Simulation 2 Neurons
neuronNum = 2
#Order: neuronNum, dt, stimStart, stimEnd, end, tau
sim = Simulation(neuronNum, .1, 100, 500, 1000, 20)
sim.SetWeightMatrixManual(np.array([[0,10],[-9,0]]))
sim.GraphEig(lines=False)
x = ConstCurrent(sim.t_vect, 0, [sim.t_stimStart, sim.t_stimEnd])
sim.SetCurrent(np.array([x,x]))
sim.RunSim(v0=[30,30])
sim.GraphNeuronsTime()
sim.GraphNeuronsTogether()
