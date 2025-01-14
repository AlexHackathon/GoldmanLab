# Import file management modules
import pickle

# Import data analysis modules
from skopt import gp_minimize
import random
import numpy as np
import scipy as sp
import scipy.optimize

# Importing required libraries for graphing
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pylab import *
from mpl_toolkits import mplot3d

#Importing custom modules
import Helpers.ActivationFunction as ActivationFunction
import SimulationClass

#Define various file locations
weightFileLoc = "NeuronWeights2.bin" #Stores weights from the previous simulation
eyeWeightFileLoc = "EyeWeights2.bin" #Stores the eye prediction weights from the previous simulation
tFileLoc = "T2.bin" #Stores the tonic input from the previous simulation
dataLoc = "/Users/alex/Documents/Github/GoldmanLab/Code/EmreThresholdSlope_NatNeuroCells_All (1).xls" #Location of the data for the tuning curves

#Define Simulation Parameters
eyeMin = -20
eyeMax = 20
eyeRes = 4000
maxFreq = 80
dt = .1
totalTime = 1000 #14000 final length for tau calculation

#Simulation equation values
r_star = 1
n_Ca = 1
t_f = 2000
r0 = 0
fractionDead = 1
firstHalf = True

#Weight limits to prevent single neuron representing the population
w_max = 1
w_min = -.5
def TauSim(parameterArray):
    """parameterArray: array of length 3 with [f, P0, t_s]"""
    myF = parameterArray[0]
    myP0 = parameterArray[1]
    myT_s = parameterArray[2]
    print(myF, myP0, myT_s)
    # Initialize Fit
    myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, myP0, myF, t_f)
    myNonlinearityNoR = lambda r_vect: ActivationFunction.SynapticFacilitationNoR(r_vect, myP0, myF, t_f)
    sim = SimulationClass.Simulation(dt, totalTime, myT_s, maxFreq, eyeMin, eyeMax, eyeRes, myNonlinearity,
                                     myNonlinearityNoR, dataLoc)
    sim.SetFacilitationValues(n_Ca, r_star, myF, t_f, myP0, r0)
    sim.FitWeightMatrixExclude(weightFileLoc, eyeWeightFileLoc, tFileLoc,
                               lambda n: sim.BoundQuadrants(n, wMax=w_max, wMin=w_min))  # (-.5, .5) works
    total = 0
    num = 0
    for e in range(len(sim.eyePos)):
        if e % 1000 == 0:
            print(e)
            myDead = SimulationClass.GetDeadNeurons(1, firstHalf, sim.neuronNum)
            myTau = sim.RunSimF(startIdx=e, dead=myDead)[-1]
            total = total + myTau
            num = num + 1
    #Return negative because we are trying to minimize the negative time constant (make it big)
    print(-total/num)
    return 3000-total/num

from skopt.plots import plot_convergence
res = pickle.load(open("BayesGraphResults.bin", "rb"))

#Start the minimization
calculate = True
if calculate:
    res = gp_minimize(func=TauSim,
        dimensions=[(0.001,1.00),(0.001,1.00),(0.01,100.00)],
        acq_func="EI",
        n_calls=50,         # the number of evaluations of f (15)
        n_initial_points=5,  # the number of random initialization points (5)
        noise=0.1**2,       # the noise level (optional)
        random_state=1234)
    try:
        pickle.dump(res, open("BayesGraphResults.bin", "wb"))
    except:
        print("Couldn't write to file")

# Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# creating the heatmap
myTau = res["func_vals"]
print(myTau)
print(res["x_iters"])
myF = [i[0] for i in  res["x_iters"]]
print(np.shape(myF))
myP0 = [i[1] for i in  res["x_iters"]]
print(np.shape(myP0))
myTauS = [i[2] for i in  res["x_iters"]]
print(np.shape(myTauS))

# setting color bar
color_map = cm.ScalarMappable(cmap="inferno")
color_map.set_array(myTau)

img = ax.scatter(myF, myP0, myTauS, c=myTau)
#img = ax.scatter(myX, myY, myZ)
plt.colorbar(color_map, ax=ax)

# adding title and labels
ax.set_title("Bayesian Trace of Simulation Parameter Values and their Time Constants")
ax.set_xlim()
ax.set_xlabel('f')
ax.set_ylabel('P0')
ax.set_zlabel('t_s')
print("Hi")
plt.show()
