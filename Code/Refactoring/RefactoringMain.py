import numpy as np
import matplotlib.pyplot as plt
import FacilitationSim as FacSim
import Refactoring.SimSupport as SimSupport
import Helpers.Bound as Bound
import SupplementalMaterialsGraphs as smg
import pickle
import pandas as pd
import TestTauCalculator as tc

#Define Simulation Parameters
eyeMin = -20
eyeMax = 20
eyeRes = 4000
maxFreq = 80
totalTime = 2000 #14000 final length for tau calculation

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

print(sim.eyePos[0], sim.eyePos[1000], sim.eyePos[2000], sim.eyePos[3000])

w_min = -.005
w_max = .1
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
#smg.SupplementalGraphsFacilitation(sim)

"""for e in range(len(sim.eyePos)):
    if(e%100==0):
        #eyePos, rVect, tau = sim.RunSimFBothDead(timeToKill, startIdx=e, dead=[])
        eyePos, rVect, tauParams = sim.RunSimFBothDead(timeToKill, startIdx=e, dead=[])
        A = tauParams[0]
        B = tauParams[1]
        t1 = tauParams[2]
        t2 = tauParams[3]
        #tauVect.append(tau)
        if e%1000==0:
            print(str(e))
            plt.plot(sim.t_vect, eyePos)
            fitEyePos = [A * np.power(np.e, -sim.t_vect[i]/t1) + B * np.power(np.e, -sim.t_vect[i]/t2) for i in range(len(sim.t_vect))]
            fitEyePos = np.array(fitEyePos)
            fitEyePos = fitEyePos + eyePos[-1]
            plt.plot(sim.t_vect, fitEyePos, label = "A: " + str(A) + " B: " + str(B) + " t1: " + str(t1) + " t2: " + str(t2))
            plt.ylim(([eyeMin,eyeMax]))
            plt.legend()
plt.xlabel("Time [ms]")
plt.ylabel("Eye position [degrees]")
plt.suptitle("Eye Position After Half Lesioning of Both Sides")
plt.show()"""

#Inactivate by lesioning completely
"""for e in range(len(sim.eyePos)):
    if e%1000==0:
        eyePos, rVect, tauParams, kIdx = sim.RunSimF(timeToKill, startIdx=e, dead=SimSupport.GetDeadNeurons(0,True,sim.neuronNum))
        plt.plot(sim.t_vect, eyePos)

        A = tauParams[0]
        B = tauParams[1]
        t1 = tauParams[2]
        t2 = tauParams[3]
        fitEyePos = [A * np.power(np.e, -t / t1) + B * np.power(np.e, -t / t2) for t in
                     sim.t_vect[kIdx:]-sim.t_vect[kIdx]]
        fitEyePos = np.array(fitEyePos)
        fitEyePos = fitEyePos + eyePos[-1]
        plt.plot(sim.t_vect[kIdx:], fitEyePos, label="A: " + str(round(A,2)) + " B: " + str(round(B,2)) + " t1: " + str(round(t1,3)) + " t2: " + str(round(t2,3)))
        #plt.ylim(([eyeMin,eyeMax]))
        plt.legend()
plt.show()"""

sampleRate = 1000
guess = (5,5,50, 1000)
functionFrac = [.8,.5,.2,.01]
sampledEyePos = []
eIdx = 0

A = np.zeros((len(functionFrac),len(sim.eyePos//sampleRate)))
B = np.zeros((len(functionFrac),len(sim.eyePos//sampleRate)))
t1 = np.zeros((len(functionFrac),len(sim.eyePos//sampleRate)))
t2 = np.zeros((len(functionFrac),len(sim.eyePos//sampleRate)))

for e in range(len(sim.eyePos)):
    if e%sampleRate==0:
        print(e)
        sampledEyePos.append(e)
        #Normal no killing
        eyePos0, rVect0 = sim.RunSimF(timeToKill, startIdx=e)
        plt.plot(sim.t_vect, eyePos0,color='g', label='Healthy Network')

        #Normal with one side killed
        #eyePos1, rVect1 = sim.RunSimF(timeToKill, startIdx=e, dead=SimSupport.GetDeadNeurons(1, True, sim.neuronNum))
        #plt.plot(sim.t_vect, eyePos1,color='r')
        #Calculate

        #Normal with whole network weakened
        #eyePos2, rVect2, kIdx2 = sim.RunSimFWeaken(timeToKill, startIdx=e,weakFrac=.50) #.95-.98 works
        #plt.plot(sim.t_vect, eyePos2,color='b')

        #Normal with half network weakened
        for i in range(len(functionFrac)):
            eyePos3, rVect3, kIdx3 = sim.RunSimFWeakenSide(timeToKill, startIdx=e, weakFrac=functionFrac[i])
            shiftedTime = sim.t_vect[kIdx3:]-sim.t_vect[kIdx3] #Sets t=0 to the start of the decay
            adjustedEyePos = eyePos3[kIdx3:]-eyePos3[-1] #Shifts the eye position down so steady state is 0
            plt.plot(sim.t_vect, eyePos3, color='r', label='Weakened Lesion')
            tauParams = [np.nan, np.nan, np.nan, np.nan]
            try:
                tauParams = tc.MyCurveFitter(shiftedTime, adjustedEyePos, guess)
                print(tauParams)
            except:
                continue
            A[i, eIdx] = tauParams[0]
            B[i, eIdx] = tauParams[1]
            t1[i, eIdx] = tauParams[2]
            t2[i, eIdx] = tauParams[3]

            if tauParams[0] != np.nan:
                y = [tauParams[0]*np.power(np.e, -t/tauParams[2]) + tauParams[1]*np.power(np.e, -t/tauParams[3]) + eyePos3[-1] for t in sim.t_vect]
                plt.plot(sim.t_vect, y, color='purple', label='Exponential Fit')
            else:
                y = np.ones(len(sim.t_vect)) * eyePos3[0]
                plt.scatter(sim.t_vect, y, color='purple', label='Exponential Fit')
        print(eIdx)
        eIdx = eIdx + 1
        #Lesioned
        #shiftedTime = sim.t_vect[kIdx:] - sim.t_vect[kIdx]
        #croppedEyePos = eyePos[kIdx:]
        #tauParams = tc.MyCurveFitter(shiftedTime, croppedEyePos, (1,20,39, 500))
        #A = tauParams[0]
        #B = tauParams[1]
        #t1 = tauParams[2]
        #t2 = tauParams[3]
        #fitEyePos = [A * np.power(np.e, -t / t1) + B * np.power(np.e, -t / t2) for t in
        #             sim.t_vect[kIdx:]-sim.t_vect[kIdx]]
        #fitEyePos = np.array(fitEyePos)
        #fitEyePos = fitEyePos + eyePos[-1]
        #plt.plot(sim.t_vect[kIdx:], fitEyePos, label="A: " + str(round(A,2)) + " B: " + str(round(B,2)) + " ts: " + str(round(t1,3)) + " tf: " + str(round(t2,3)))
        #plt.ylim(([eyeMin,eyeMax]))
        #plt.legend()

        #Non-lesioned
        #plt.plot(sim.t_vect, eyePos0)
plt.xlabel("Time (ms)")
plt.ylabel("Eye Position")
plt.legend()
plt.show()
pickle.dump((A,B,t1,t2), open("FitParams.bin", "wb"))
#Old tau code with single exponential
"""
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
plt.show()"""