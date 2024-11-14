import random
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize

import Helpers.ActivationFunction as ActivationFunction
import Helpers.CurrentGenerator
import TuningCurves
import Helpers.Bound
import Helpers.GraphHelpers
import SimulationClass

weightFileLoc = "NeuronWeights.csv"
eyeWeightFileLoc = "EyeWeights.csv"
dataLoc = "/Users/alex/Documents/Github/GoldmanLab/Code/EmreThresholdSlope_NatNeuroCells_All (1).xls"

#Define Simulation Parameters
"""print("Running Main")
P0 =.1
f = 1/100
myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitationCaRELU(r_vect, P0, f, 500, rThresh=0)
overlap = 5 #Degrees in which both sides of the brain are active
neurons = 100 #Number of neurons simulated
dt = .01
#self, neuronNum, dt, end, tau, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, nonLinearityFunction, fileLocation):
sim = SimulationClass.Simulation(neurons, dt, 2000, 20, 150, -25, 25, 5000, myNonlinearity, dataLoc)
#Graph tuning curves
sim.PlotTargetCurves()
#Set weight matrix
mode = input("How should the weight matrix be assigned: ")
if mode == "0":
    try:
        #Read the file for weights
        sim.SetWeightMatrixRead(weightFileLoc, eyeWeightFileLoc)
    except:
        #Run the fitting algorithm anyway
        print("Refitting the weights")
        sim.FitWeightMatrixExclude(weightFileLoc, eyeWeightFileLoc, sim.BoundQuadrants)
    print(sim.w_mat)
elif mode == "1":
    #Run the fitting algorithm and write the weights to the file
    sim.FitWeightMatrixExclude(weightFileLoc, eyeWeightFileLoc, sim.BoundQuadrants)
else:
    print("Invalid input")
    quit()"""
#*****BREAK
#Define Simulation
eyeMin = -25
eyeMax = 25
eyeRes = 5000
maxFreq = 150
totalTime = 2000

r_star = 11
n_Ca = 2.4
P0=.1
f = 1/100
neurons = 100
dt = .1
t_f = 500
t_s = 50
r0 = 10
#myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, P0, f, t_f)
myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitationCa(r_vect, P0, f, t_f,n_Ca,r_star)
#myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitationCaRELU(r_vect, P0, f, t_f, r0)
sim = SimulationClass.Simulation(neurons, dt, totalTime, t_s, maxFreq, eyeMin, eyeMax, eyeRes, myNonlinearity, dataLoc)
sim.SetFacilitationValues(n_Ca, r_star, f, t_f, P0, r0)
#Graph tuning curves
#sim.PlotTargetCurves()
#Set weight matrix
mode = "1"#input("How should the weight matrix be assigned: ")
w_max = np.inf
w_min = -np.inf
if mode == "0":
    try:
        #Read the file for weights
        sim.SetWeightMatrixRead(weightFileLoc, eyeWeightFileLoc)
    except:
        #Run the fitting algorithm anyway
        print("Refitting the weights")
        sim.FitWeightMatrixExclude(weightFileLoc, eyeWeightFileLoc, lambda n: sim.BoundQuadrants(n, wMax=w_max, wMin=w_min)) #(-.5, .5) works
    print(sim.w_mat)
elif mode == "1":
    #Run the fitting algorithm and write the weights to the file
    sim.FitWeightMatrixExclude(weightFileLoc, eyeWeightFileLoc, lambda n: sim.BoundQuadrants(n, wMax=w_max, wMin=w_min))
else:
    print("Invalid input")
    quit()

sim.PlotFixedPointsOverEyePosRate(range(len(sim.r_mat)))
plt.show()

sim.GraphWeightMatrix()
#Run the simulation within the simulation class (Choose and store)
#Graphing Code
for e in range(len(sim.eyePos)):
    if e%1000 == 0:
        print(e)
        #mySimRes = sim.RunSimFCaRELU(startIdx=e)
        mySimRes = sim.RunSimFCa(-1, startIdx=e, dead=SimulationClass.GetDeadNeurons(1, sim.neuronNum, True))
        plt.plot(sim.t_vect, mySimRes[0])
        plt.ylim([eyeMin,eyeMax])
plt.show()