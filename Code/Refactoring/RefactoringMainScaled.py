import numpy as np
import matplotlib.pyplot as plt
import FacilitationSim as FacSim
import Refactoring.SimSupport as SimSupport
import Helpers.Bound as Bound
import SupplementalMaterialsGraphs as smg
import pickle

#Define Simulation Parameters
eyeMin = -20
eyeMax = 20
eyeRes = 4000
maxFreq = 80
totalTime = 1000 #14000 final length for tau calculation

dt = .01
P0=.01
f = 1
t_f = 500
t_s = 50


timeToKill = 100
fractionDead = 1
firstHalf = True

dataLoc = "../EmreThresholdSlope_NatNeuroCells_All (1).xls"

#Define Facilitation Simulation
parameters = FacSim.FacilitationParameters(dt, totalTime,t_s, maxFreq, eyeMin, eyeMax, eyeRes, P0, f, t_f)
sim = FacSim.Simulation(parameters, dataLoc)

w_min = -.005
w_max = 100
bounds = [Bound.BoundQuadrants(n, w_min, w_max, sim.neuronNum) for n in range(sim.neuronNum)]

calc = True
dump = False
fileName = "DebugDump.bin"
if calc:
    sim.w_mat, sim.T = SimSupport.FitWeightMatrixExclude(sim.r_mat, sim.r_mat_neg, sim.f, bounds)
    sim.FitPredictorNonlinearSaturation()
    if dump:
        pickle.dump((sim.w_mat, sim.T, sim.predictW, sim.predictT), open(fileName, "wb"))
else:
    sim.w_mat, sim.T, sim.predictW, sim.predictT = pickle.load(open(fileName, "rb"))
smg.SupplementalGraphsFacilitation(sim)

for e in range(len(sim.eyePos)):
    if e%1000==0:
        eyePos, rVect, tauVect = sim.RunSimF(timeToKill, startIdx=e, dead=SimSupport.GetDeadNeurons(1,True,sim.neuronNum))
        plt.plot(sim.t_vect, eyePos)
        plt.ylim(([eyeMin,eyeMax]))
plt.show()
tauAvg = []
numPoints = 20
for tauF in np.linspace(1,2000,numPoints):
    # Define Facilitation Simulation
    parameters = FacSim.FacilitationParameters(dt, totalTime, t_s, maxFreq, eyeMin, eyeMax, eyeRes, P0, f, tauF)
    sim = FacSim.Simulation(parameters, dataLoc)

    w_min = -.005
    w_max = 100
    bounds = [Bound.BoundQuadrants(n, w_min, w_max, sim.neuronNum) for n in range(sim.neuronNum)]

    sim.w_mat, sim.T = SimSupport.FitWeightMatrixExclude(sim.r_mat, sim.r_mat_neg, sim.f, bounds)

    print("Tau f: " + str(tauF))
    sim.FitPredictorNonlinearSaturation()

    tauVect = []
    for e in range(len(sim.eyePos)):
        if e % 1000 == 0:
            eyePos, rVect, tau = sim.RunSimF(timeToKill, startIdx=e,
                                                 dead=SimSupport.GetDeadNeurons(1, True, sim.neuronNum))
            tauVect.append(tau)
    tauAvg.append(np.average(tauVect))
plt.plot(np.linspace(1,2000,numPoints), tauAvg)
plt.suptitle("Time Constants for Varying Synaptic Facilitation Time Constants")
plt.xlabel("Tau F")
plt.ylabel("Tau Decay")
plt.show()

