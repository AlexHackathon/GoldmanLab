import FacilitationSim as FacSim
import Refactoring.SimSupport as SimSupport
import Helpers.Bound as Bound
import matplotlib.pyplot as plt
#Define Simulation Parameters
eyeMin = -20
eyeMax = 20
eyeRes = 4000
maxFreq = 80
totalTime = 1000 #14000 final length for tau calculation

dt = .01
P0=.01
f = 1
t_f = 2000
t_s = 50

parameters = FacSim.FacilitationParameters(dt, totalTime,t_s, maxFreq, eyeMin, eyeMax, eyeRes, P0, f, t_f)

timeToKill = 100
fractionDead = 1
firstHalf = True

dataLoc = "../EmreThresholdSlope_NatNeuroCells_All (1).xls"

#Define Facilitation Simulation
sim = FacSim.Simulation(parameters, dataLoc)

w_min = -.005
w_max = 100
bounds = [Bound.BoundQuadrants(n, w_min, w_max, sim.neuronNum) for n in range(sim.neuronNum)]

sim.w_mat, sim.T = SimSupport.FitWeightMatrixExclude(sim.r_mat, sim.r_mat_neg, sim.f, bounds)
print("MADE IT!")
sim.FitPredictorNonlinearSaturation()

for e in range(len(sim.eyePos)):
    if e%1000==0:
        eyePos, rVect, tauVect = sim.RunSimF(timeToKill, startIdx=e, dead=SimSupport.GetDeadNeurons(1,True,sim.neuronNum))
        plt.plot(sim.t_vect, eyePos)
        plt.ylim(([eyeMin,eyeMax]))
plt.show()
