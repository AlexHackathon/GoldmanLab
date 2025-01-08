import pickle
import random
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
import SimulationClass

from skopt import gp_minimize
weightFileLoc = ""
eyeWeightFileLoc = ""
tFileLoc = ""
dataLoc = "/Users/alex/Documents/Github/GoldmanLab/Code/EmreThresholdSlope_NatNeuroCells_All (1).xls"

#Define Simulation Parameters
eyeMin = -20
eyeMax = 20
eyeRes = 4000
maxFreq = 80
totalTime = 1000 #14000 final length for tau calculation

r_star = 1
n_Ca = 1


neurons = 100
dt = .1
t_f = 2000
r0 = 0

fractionDead = 1
firstHalf = True

w_max = 1
w_min = -.5
def MaximizeFunction(parameterArray):
    """parameterArray: array of length 3 with [f, P0, t_s]"""
    myF = parameterArray[0]
    myP0 = parameterArray[1]
    myT_s = parameterArray[2]
    # Initialize Fit
    myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, myP0, myF, t_f)
    myNonlinearityNoR = lambda r_vect: ActivationFunction.SynapticFacilitationNoR(r_vect, myP0, myF, t_f)
    sim = SimulationClass.Simulation(neurons, dt, totalTime, myT_s, maxFreq, eyeMin, eyeMax, eyeRes, myNonlinearity,
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
    return -total/num

#Start the minimization
res = gp_minimize(MaximizeFunction
    [(0,1),(0,1),(0,100)],
    acq_func="EI",
    n_calls=15,         # the number of evaluations of f
    n_random_starts=5,  # the number of random initialization points
    noise=0.1**2,       # the noise level (optional)
    random_state=1234)
try:
    pickle.dump(res, open("BayesGraphResults.bin", "wb"))
except:
    print("Couldn't write to file")
#Use x_iters func_vals to plot the diagram
