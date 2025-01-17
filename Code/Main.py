import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy as sp
import scipy.optimize

import Helpers.ActivationFunction as ActivationFunction
import Helpers.CurrentGenerator
import TuningCurves
import Helpers.Bound
import Helpers.GraphHelpers
import TestTauCalculator as tc
import SimulationClass
import pickle

dataLoc = "EmreThresholdSlope_NatNeuroCells_All (1).xls"
weightFileLoc = "NeuronWeights.bin"
eyeWeightFileLoc = "EyeWeights.bin"
tFileLoc = "T.bin"

#Define Simulation Parameters
eyeMin = -20
eyeMax = 20
eyeRes = 4000
maxFreq = 80
totalTime = 1000 #14000 final length for tau calculation

r_star = 1
n_Ca = 1

#0.4782558584049041 0.9835457913046722 43.27910201786918
P0=.001
f = 1
dt = .1
t_f = 2000
t_s = 100
r0 = 0

fractionDead = 1
firstHalf = True

simType = "F"
myNonlinearity = None
myNonlinearityNoR = None
if(simType == "CaRELU"):
    myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitationCaRELU(r_vect, P0, f, t_f, r0)
    myNonlinearityNoR = lambda r_vect: ActivationFunction.SynapticFacilitationCaRELUNoR(r_vect, P0, f, t_f, r0)
elif(simType == "Ca"):
    myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitationCa(r_vect, P0, f, t_f, n_Ca, r_star)
    myNonlinearityNoR = lambda r_vect: ActivationFunction.SynapticFacilitationCaNoR(r_vect, P0, f, t_f, r0)
elif(simType == "F"):
    myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, P0, f, t_f)
    myNonlinearityNoR = lambda r_vect: ActivationFunction.SynapticFacilitationNoR(r_vect, P0, f, t_f)
else:
    print("Sim not found")
if not(myNonlinearity == None) and not(myNonlinearityNoR == None):
    sim = SimulationClass.Simulation(dt, totalTime, t_s, maxFreq, eyeMin, eyeMax, eyeRes, myNonlinearity, myNonlinearityNoR,dataLoc)
else:
    quit()
sim.SetFacilitationValues(n_Ca, r_star, f, t_f, P0, r0)
#Set weight matrix
mode = input("How should the weight matrix be assigned: ") #read or calc
w_max = .05
w_min = -.0005
if mode == "read":
    try:
        #Read the file for weights
        sim.ReadWeightMatrix(weightFileLoc, eyeWeightFileLoc,tFileLoc)
    except Exception as e:
        print(e)
        #Run the fitting algorithm anyway
        print("Refitting the weights")
        sim.FitWeightMatrixExclude(weightFileLoc, eyeWeightFileLoc, tFileLoc, lambda n: sim.BoundQuadrants(n, wMax=w_max, wMin=w_min)) #(-.5, .5) works
        quit()
elif mode == "calc":
    #Run the fitting algorithm and write the weights to the file
    sim.FitWeightMatrixExclude(weightFileLoc,eyeWeightFileLoc, tFileLoc, lambda n: sim.BoundQuadrants(n, wMax=w_max, wMin=w_min))
else:
    print("Invalid input")
    quit()
#Graphing
grid = GridSpec(6,6)
fig = plt.figure()

#Synaptic Activation curves
ax1=fig.add_subplot(grid[0,0])
r,s = sim.SynapticActivationCurves()
plt.plot(r,s)

ax2=fig.add_subplot(grid[0,1])
r,s = sim.SynapticActivationCurves()
plt.plot(r,s)

#Presynaptic Neuron Tuning Curves
#First half
ax3=fig.add_subplot(grid[1,0])
for n in sim.r_mat[:sim.neuronNum//2,:]:
    plt.plot(sim.eyePos, n)

#Second half
ax4=fig.add_subplot(grid[1,1])
for n in sim.r_mat[sim.neuronNum//2:,:]:
    plt.plot(sim.eyePos, n)

#Presynaptic neuron activations
#First half
ax5=fig.add_subplot(grid[2,0])
for n in sim.r_mat[:sim.neuronNum//2,:]:
    plt.plot(sim.eyePos, sim.f(n))

#Second half
ax6=fig.add_subplot(grid[2,1])
for n in sim.r_mat[sim.neuronNum//2:,:]:
    plt.plot(sim.eyePos, sim.f(n))

#For Single Neuron
targetNeuron = 5
#Individual Inputs With Weights
#Excitatory
ax7=fig.add_subplot(grid[3,0])
rContributionE = np.zeros((sim.eyeRes,sim.neuronNum//2))
s = sim.f(sim.r_mat[:sim.neuronNum//2,:])
for i in range(len(s)):
    for j in range(len(s[i])):
        s[i,j] = s[i,j] * sim.w_mat[targetNeuron,i]
for s_n in s:
    plt.plot(sim.eyePos,s_n)

#Inhibitory
ax8=fig.add_subplot(grid[3,1])
rContributionI = np.zeros((sim.eyeRes,sim.neuronNum//2))
s = sim.f(sim.r_mat[sim.neuronNum//2:,:])
for i in range(len(s)):
    for j in range(len(s[i])):
        s[i,j] = s[i,j] * sim.w_mat[targetNeuron,i]
for s_n in s:
    plt.plot(sim.eyePos,s_n)

#Total Inputs
#Excitatory
ax9=fig.add_subplot(grid[4,0])
s = sim.f(sim.r_mat[:sim.neuronNum//2,:])
totalInputA = np.zeros((eyeRes,))
for e in range(sim.eyeRes):
    totalInputA[e] = np.dot(s[:,e],sim.w_mat[targetNeuron,:sim.neuronNum//2])
plt.plot(sim.eyePos,totalInputA)

#Inhibitory
ax10=fig.add_subplot(grid[4,1])
s = sim.f(sim.r_mat[sim.neuronNum//2:,:])
totalInputB = np.zeros((eyeRes,))
for e in range(sim.eyeRes):
    totalInputB[e] = np.dot(s[:,e],sim.w_mat[targetNeuron,sim.neuronNum//2:])
plt.plot(sim.eyePos,totalInputB)

#Total All Inputs
ax11=fig.add_subplot(grid[5,:1])
total=totalInputA+totalInputB
plt.plot(sim.eyePos, total)
plt.plot(sim.eyePos, sim.r_mat[targetNeuron])
plt.plot(sim.eyePos, sim.T[targetNeuron]*np.ones((len(sim.eyePos,))))
plt.plot(sim.eyePos, total + sim.T[targetNeuron]*np.ones((len(sim.eyePos,))))

ax12=fig.add_subplot(grid[:3,2:])
#Run the simulation within the simulation class (Choose and store)
#Graphing Code
if(simType == "CaRELU"):
    for e in range(len(sim.eyePos)):
        if e % 1000 == 0:
            print(e)
            mySimRes = sim.RunSimFCaRELUVariable(startIdx=e)
            plt.plot(sim.t_vect, mySimRes[0], color="green")
            myDead = SimulationClass.GetDeadNeurons(1, firstHalf, sim.neuronNum)
            mySimRes2 = sim.RunSimFCaRELUVariable(startIdx=e, dead=myDead)
            plt.plot(sim.t_vect, mySimRes2[0], color="red")
            plt.ylim([eyeMin,eyeMax])
elif(simType == "Ca"):
    for e in range(len(sim.eyePos)):
        if e % 1000 == 0:
            print(e)
            #mySimRes = sim.RunSimFCa(startIdx=e, fixT_f=500)
            mySimRes = sim.RunSimFCa(startIdx=e)
            plt.plot(sim.t_vect, mySimRes[0], color="green")
            myDead = SimulationClass.GetDeadNeurons(1, firstHalf, sim.neuronNum)
            #mySimRes2 = sim.RunSimFCa(startIdx=e, dead=myDead, fixT_f=500)
            mySimRes2 = sim.RunSimFCa(startIdx=e, dead=myDead)
            plt.plot(sim.t_vect, mySimRes2[0], color="red")
            plt.ylim([eyeMin,eyeMax])
elif(simType == "F"):
    for e in range(len(sim.eyePos)):
        if e % 1000 == 0:
            print(e)
            mySimRes = sim.RunSimF(startIdx=e)
            plt.plot(sim.t_vect, mySimRes[0], color="green")
            myDead = SimulationClass.GetDeadNeurons(1, firstHalf, sim.neuronNum)
            mySimRes2 = sim.RunSimF(startIdx=e, dead=myDead)
            plt.plot(sim.t_vect, mySimRes2[0], color="red")
            plt.ylim([eyeMin, eyeMax])
else:
    print("Sim not found")
if simType == "CaRELU":
    plt.title("Simulation with r0 = " + str(r0))
elif simType == "Ca":
    plt.title("Simulation with r*=" + str(r_star) + " n=" + str(n_Ca))
ax13=fig.add_subplot(grid[3:,2:])
sim.GraphWeightMatrix()
plt.show()
