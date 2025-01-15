# Import file management modules
import pickle
import multiprocessing

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


def HelperFit(startN, endN, X, bounds, r_mat_neg, conn):
    halfSol = []
    for myN in range(startN, endN):
        r = r_mat_neg[myN]
        solution = None
        if bounds != None:
            bounds = bounds[myN]
            solution = scipy.optimize.lsq_linear(X, r, bounds)
        else:
            solution = scipy.optimize.lsq_linear(X, r)
        halfSol.append(solution)
    conn.send(halfSol)
    conn.close()
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
    #Set up all variables before initiating parallel processing (Needed for pickling)
    X = np.ones((len(sim.eyePos), len(sim.r_mat) + 1))
    for i in range(len(X)):
        for j in range(len(X[0]) - 1):
            X[i, j] = sim.f(sim.r_mat[j, i])
    bounds = [sim.BoundQuadrants(n, w_min, w_max) for n in range(sim.neuronNum)]
    if __name__ == '__main__':
        mainConn1, firstConn = multiprocessing.Pipe()
        mainConn2, secondConn = multiprocessing.Pipe()
        firstHalfProcess = multiprocessing.Process(target=HelperFit, args=(0, sim.neuronNum // 2, X, bounds, sim.r_mat_neg, firstConn))
        secondHalfProcess = multiprocessing.Process(target=HelperFit, args=(sim.neuronNum // 2, sim.neuronNum, X, bounds, sim.r_mat_neg, secondConn))
        firstHalfProcess.start()
        secondHalfProcess.start()
        firstHalfProcess.join()
        secondHalfProcess.join()
        firstHalfResult = mainConn1.recv()
        secondHalfResult = mainConn2.recv()
        solution = np.concatenate((firstHalfResult,secondHalfResult))
        sim.w_mat = solution[:-1]
        sim.T = solution[-1]
    sim.WriteWeightMatrix(sim.w_mat, weightFileLoc)
    sim.WriteWeightMatrix(sim.T, tFileLoc)
    sim.FitPredictorNonlinearSaturation(eyeWeightFileLoc)
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

#Start the minimization
calculate = False
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
else:
    res = pickle.load(open("BayesGraphResults.bin", "rb"))

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

bestRes = res['x']
bestF = bestRes[0]
bestP0 = bestRes[1]
bestTs = bestRes[2]
myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, bestP0, bestF, t_f)
myNonlinearityNoR = lambda r_vect: ActivationFunction.SynapticFacilitationNoR(r_vect, bestP0, bestF, t_f)
sim = SimulationClass.Simulation(dt, totalTime, bestTs, maxFreq, eyeMin, eyeMax, eyeRes, myNonlinearity,
                                     myNonlinearityNoR, dataLoc)
sim.FitWeightMatrixExclude(weightFileLoc, eyeWeightFileLoc, tFileLoc, lambda n: sim.BoundQuadrants(n, w_max, w_min))
myDead = SimulationClass.GetDeadNeurons(1, firstHalf, sim.neuronNum)
for e in range(len(sim.eyePos)):
    if e % 1000 == 0:
        print(e)
        myDead = SimulationClass.GetDeadNeurons(1, firstHalf, sim.neuronNum)
        eyePos = sim.RunSimF(startIdx=e, dead=myDead)[0]
        plt.plot(sim.t_vect, eyePos)
plt.show()