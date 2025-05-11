#region import statements
import numpy as np
import matplotlib.pyplot as plt
import pandas

import FacilitationSim as FacSim
import Refactoring.SimSupport as SimSupport
import Helpers.Bound as Bound
import SupplementalMaterialsGraphs as smg
import pickle
import pandas as pd
import TestTauCalculator as tc
import scipy.optimize as skopt
import os
import re
import time
#endregion
#region Input Code
def CheckInputs(prompt, acceptableAnswers):
    ans = input(prompt)
    if ans in acceptableAnswers:
        return ans
    else:
        while ans not in acceptableAnswers:
            print("Answer is not acceptable.")
            ans = input(prompt)
        return ans
#endregion
#region Simulation Parameters

#Tuning curve params
eyeMin = -20
eyeMax = 20
eyeRes = 1000
maxFreq = 80

#Simulation params
totalTime = 5000
dt = .01
P0=.01
f = .02 * .001
t_f = 6000
t_s = 50

#Lesioning params
timeToKill = 100
fractionDead = 1
firstHalf = True

#Data location
dataLoc = "../EmreThresholdSlope_NatNeuroCells_All (1).xls"
#endregion
#Define Facilitation Simulation
#Instantiate the simulation
parameters = FacSim.FacilitationParameters(dt, totalTime,t_s, maxFreq, eyeMin, eyeMax, eyeRes, P0, f, t_f) #Useless revert to 20 params
sim = FacSim.Simulation(parameters, dataLoc)
#region Fit code
w_min = -np.inf #Minimum inhibitory weight
w_max = np.inf #Maximum excitatory weight
bounds = [Bound.BoundQuadrants(n, w_min, w_max, sim.neuronNum) for n in range(sim.neuronNum)]

fileName = "Weights.bin"
#Should you calculate the weights again or should you just read them from the debug dump?
calcWeight = False
print("Calculate Weight: " + str(calcWeight))
# Should you dump the weights into the Weights.bin?
saveWeight = True
print("Save Weight: " + str(saveWeight))
if calcWeight:
    sim.w_mat, sim.T = SimSupport.FitWeightMatrixExclude(sim.r_mat, sim.r_mat_neg, sim.f, bounds)
    sim.FitPredictorNonlinearSaturation()
    if saveWeight:
        pickle.dump((sim.w_mat, sim.T, sim.predictW, sim.predictT), open(fileName, "wb"))
else:
    #Read the fit from a previous run of the simulation
    sim.w_mat, sim.T, sim.predictW, sim.predictT = pickle.load(open(fileName, "rb"))

#Do you want to graph the fit and visualize the weight matrix?
supGraph = False
print("Draw supplemental: " + str(supGraph))
if supGraph:
    smg.SupplementalGraphsFacilitation(sim)
#endregion

#region Complete Simulation

#Do you want to run the basic simulation?
runBasic = False
print("Run Sim Facilitation Intact: " + str(runBasic))
# Do you want to run the visualizations?
visBasic = True
print("Visualize Intact: " + str(visBasic))
if runBasic:
    #Run the simulation for 10 eye positions
    numEyePositions = 5
    for e in np.linspace(0,len(sim.eyePos)-1,5, dtype=int):
        eyePositions, firingRates = sim.RunSimF(timeAtKill=timeToKill, startIdx=e)
        # Run the simulation with pulses
            #Missing
        #Complete Simulation Visualization
        if visBasic:
            plt.plot(sim.t_vect, eyePositions)
        plt.suptitle("Synaptic Facilitation Simulation")
        plt.xlabel("Time (ms)")
        plt.ylabel("Eye position (degrees)")
    plt.show()
    # Graph the results over time with added pulses of input
    # Missing
#endregion

#region Lesion Simulation

epdf = None #Eye position data frame
if os.path.exists("SimulationData/SimulationDataframe.csv"):
    epdf = pd.read_csv("SimulationData/SimulationDataframe.csv", low_memory=False)
else:
    data = {"eyePos": [],
            "weakFrac": [],
            "simType": [],
            "simulation": [],
            "f": [],
            "t_f": [],
            "t_s": [],
            "killIdx": [],
            "A": [],
            "t_fast": [],
            "B": [],
            "t_slow": [],
            "offset": []}
    epdf = pd.DataFrame(data)
#How many eye positions should it simulate?
numberEyePositions = 2
print("Running lesioning experiments.")
runSimOneSide = False
print("One sided inactivation: " + str(runSimOneSide))
runSimBothSides = True
print("Two sided inactivation: " + str(runSimBothSides))
print("Simulating lesion at these eye positions:")
for e in np.linspace(0,len(sim.eyePos)-1,numberEyePositions, dtype=int):
    print(sim.eyePos[e])
    # Run the simulation for 10 eye positions w/ 10%, 50%, and 90% inactivation on one side
    #Should we run a single sided lesion experiment?
    weakFracType = [.9, .5]
    if runSimOneSide:
        for weakFrac in weakFracType:
            print(weakFrac)
            cA = epdf["eyePos"] == sim.eyePos[e]
            cB = epdf["weakFrac"] == weakFrac
            cC = epdf["f"] == f
            cD = epdf["t_f"] == t_f
            cE = epdf["t_s"] == t_s
            cF = epdf["simType"] == "one"
            condition = cA & cB & cC & cD & cE & cF
            rowExists = not epdf[condition].empty
            if not rowExists:
                fileLocation = "SimulationData/Data:"+"E:"+str(round(sim.eyePos[e],2)) +"wf:"+str(weakFrac)+"f:"+str(f)+"t_f:"+str(t_f)+"t_s"+str(t_s)+"type:one"
                if os.path.exists(fileLocation):
                    print("File already exists: " + fileLocation)
                    #myInput = input("Overwrite? :")
                    #if myInput=="t":
                    #    pickle.dump(eyePositions, open(fileLocation, "wb"))
                else:
                    eyePositions, _, firstIdx = sim.RunSimFWeakenSide(timeToKill, e, weakFrac=weakFrac)
                    pickle.dump((sim.t_vect, eyePositions), open(fileLocation, "wb"))
                    newRow = pd.DataFrame({"eyePos" : [sim.eyePos[e]],
                                            "weakFrac": [weakFrac],
                                           "simType": ["one"],
                                           "simulation": [fileLocation],
                                            "f": [f],
                                            "t_f": [t_f],
                                            "t_s": [t_s],
                                            "killIdx": [firstIdx]})
                    epdf = pd.concat([epdf, newRow], ignore_index=True)
    #Should we run a two-sided lesion experiment?
    if runSimBothSides:
        for weakFrac in weakFracType:
            print(weakFrac)
            cA = epdf["eyePos"] == sim.eyePos[e]
            cB = epdf["weakFrac"] == weakFrac
            cC = epdf["f"] == f
            cD = epdf["t_f"] == t_f
            cE = epdf["t_s"] == t_s
            cF = epdf["simType"] == "both"
            condition = cA & cB & cC & cD & cE & cF
            rowExists = not epdf[condition].empty
            if not rowExists:
                fileLocation = "SimulationData/Data:"+"E:"+str(round(sim.eyePos[e],2)) +"wf:"+str(weakFrac)+"f:"+str(f)+"t_f:"+str(t_f)+"t_s"+str(t_s)+"type:both"
                if os.path.exists(fileLocation):
                    print("File already exists: " + fileLocation)
                    #myInput = input("Overwrite? :")
                    #if myInput=="t":
                    #    pickle.dump(eyePositions, open(fileLocation, "wb"))
                else:
                    eyePositions, _, firstIdx = sim.RunSimFWeaken(timeToKill, e, weakFrac=weakFrac)
                    pickle.dump((sim.t_vect, eyePositions), open(fileLocation, "wb"))
                    #epdf.loc[len(epdf)] = [sim.eyePos[e], weakFrac, fileLocation, "both", f, t_f, t_s, firstIdx,np.nan,np.nan,np.nan,np.nan,np.nan]
                    newRow = pd.DataFrame({"eyePos": [sim.eyePos[e]],
                                           "weakFrac": [weakFrac],
                                           "simulation": [fileLocation],
                                           "simType": ["both"],
                                           "f": [f],
                                           "t_f": [t_f],
                                           "t_s": [t_s],
                                           "killIdx": [firstIdx]})
                    epdf = pd.concat([epdf, newRow], ignore_index=True)
                    #epdf.append(newRow,ignore_index=True)
epdf.to_csv("SimulationData/SimulationDataframe.csv",index=False)
quit()
#endregion

#region Define Double Exponential Fitter fit_double_exponential(x,y,initial)
def exponentialNormalization(xData, yData, killIdx):
    alignedXData = xData[killIdx:] - xData[killIdx]
    cutYData = yData[killIdx:]
    return alignedXData, cutYData
def double_exponential(x, a1, tau1, a2, tau2, offset):
    """Defines the double exponential function."""
    return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + offset

def fit_double_exponential(x_data, y_data, initialGuess=None):#initial_guess, startTime):
    """Fits a double exponential function using scipy.optimize.minimize."""
    popt = None
    if initialGuess!=None:
        popt, pcov = skopt.curve_fit(double_exponential, x_data, y_data, p0=initialGuess)
    else:
        popt, pcov = skopt.curve_fit(double_exponential, x_data, y_data)
    return popt

#region Lesion Simulation Visualization
epdf = pandas.read_csv("SimulationData/SimulationDataframe.csv")
shouldFit = True
overwriteTauAnalysis = True
print("Double exponential fit eye position: " + str(shouldFit))
if shouldFit:
    #Check if you should manually input guesses
    """manualGuess = CheckInputs("Do you want to manually input initial guesses(T/F): ", ["t", "f"])
    if manualGuess.lower() == "t":
        manualGuess = True
    else:
        manualGuess = False
    print("Fitting tau at these eye positions: ")"""
    #Fit code
    for index, row in epdf.iterrows():
        print(row["A"])
        if pd.isna(row["A"]) or overwriteTauAnalysis:
            print("fitting")
            #Load the eye position trace
            simTime, simEyePos = pickle.load(open(row["simulation"],"rb"))
            simKillIdx = int(row["killIdx"])
            normTime, normEyePos = exponentialNormalization(simTime, simEyePos, 10001)
            if(normEyePos[0]<0):
                initialGuess = [-10,50,-10,500,0]
                epdf.loc[index, "A"], epdf.loc[index, "t_fast"], epdf.loc[index, "B"], epdf.loc[index, "t_slow"], \
                epdf.loc[index, "offset"] = fit_double_exponential(normTime, normEyePos, initialGuess)
            else:
                epdf.loc[index, "A"], epdf.loc[index, "t_fast"], epdf.loc[index, "B"], epdf.loc[index, "t_slow"], \
                epdf.loc[index, "offset"] = fit_double_exponential(normTime, normEyePos)
            #Calculate the new trace and fit the data
            yCalc = double_exponential(normTime, epdf.loc[index,"A"], epdf.loc[index,"t_fast"], epdf.loc[index,"B"], epdf.loc[index,"t_slow"], epdf.loc[index,"offset"])
            plt.plot(simTime, simEyePos, color="green")
            plt.plot(simTime[simKillIdx:], yCalc, color="red")
    """
    initialGuess = None
    if not manualGuess:
        guessA = (simEyePos[-1] - simEyePos[0]) * 2/3
        guessB = simEyePos[-1] - simEyePos[0] * 1/3
        guessO = simEyePos[-1]
        initialGuess = [guessA, t_s, guessB, t_f, guessO]
        print("Automatically Guessing: " + str(initialGuess))
    else:
        plt.plot(cutTime, cutEyePositions[simKillIdx:])
        plt.show()
        ans = input("Enter your guess A t1 B t2 Offset: ")
        initialGuess = re.findall(r'\d+', ans)
        while len(initialGuess) != 5:
            print("Wrong number of parameter guesses in the exponential fit.")
            ans = input("Enter your guess A t1 B t2 Offset: ")
            initialGuess = re.findall(r'\d+', ans)
    #ans = fit_double_exponential(cutTime, cutEyePositions, initialGuess, cutTime[0])
    ans = fit_double_exponential(cutTime, cutEyePositions, cutTime[0])"""
    epdf.to_csv("SimulationData/SimulationDataframe.csv", index=False)
plt.show()

mode = "currParam"
if mode == "currParam":
    cC = epdf["f"] == f
    cD = epdf["t_s"] == t_s
    cE = epdf["t_f"] == t_f
    cF = epdf["weakFrac"] == .1
    cG = epdf["simType"] == "both"
    condition = cC & cD & cE & cF & cG

    matchingRows = epdf.loc[condition]
    print("Printing matching rows")
    print(matchingRows)
    for idx,row in matchingRows.iterrows():
        print(row)
        simTime, eyePositions = pickle.load(open(row["simulation"],"rb"))
        simKillIdx  = int(row["killIdx"])
        plt.plot(simTime[simKillIdx:], eyePositions[simKillIdx:], color="green")
        A = row["A"]
        t_fast = row["t_fast"]
        B = row["B"]
        t_slow = row["t_slow"]
        fitEyePos = [A * np.power(np.e, -simTime[simKillIdx+i] / t_fast) + B * np.power(np.e, -simTime[simKillIdx+i] / t_slow) for i in
                     range(len(simTime[simKillIdx:]))]
        fitEyePos = np.array(fitEyePos)
        fitEyePos = fitEyePos + eyePositions[-1]
        plt.plot(simTime[simKillIdx:], fitEyePos,color="red")#, label="A: " + str(A) + " B: " + str(B) + " t_fast: " + str(t_fast) + " t_slow: " + str(t_slow), color="red")
        plt.ylim(([eyeMin, eyeMax]))
        plt.legend()
    plt.show()

    print("Plotting Tau Statistics")
    plt.suptitle("A (Magnitude ts) for 90% Inactivation")
    plt.hist(matchingRows["A"])
    plt.show()

    plt.suptitle("B (Magnitude tf) for 90% Inactivation")
    plt.hist(matchingRows["B"])
    plt.show()

    plt.suptitle("Fast time constant for 90% Inactivation")
    plt.hist(matchingRows["t_fast"])
    plt.xlabel("Time [ms]")
    plt.show()

    plt.suptitle("Slow time constant for 90% Inactivation")
    plt.hist(matchingRows["t_slow"])
    plt.xlabel("Time [ms]")
    plt.show()
#Calculate the error in the double exponential fit
#Missing
#endregion

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

quit()
sampleRate = 1000
guess = (10,.1,30, 3000) #(5,5,50,1000)
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
        plt.plot(sim.t_vect, eyePos0, color='g')

        #Normal with one side killed
        #eyePos1, rVect1 = sim.RunSimF(timeToKill, startIdx=e, dead=SimSupport.GetDeadNeurons(1, True, sim.neuronNum))
        #plt.plot(sim.t_vect, eyePos1,color='r')
        #Calculate

        #Normal with whole network weakened
        eyePos2, rVect2, kIdx2 = sim.RunSimFWeaken(timeToKill, startIdx=e,weakFrac=.3) #.95-.98 works
        shiftedTime = sim.t_vect[kIdx2:] - sim.t_vect[kIdx2]  # Sets t=0 to the start of the decay
        adjustedEyePos = eyePos2[kIdx2:] - eyePos2[-1]  # Shifts the eye position down so steady state is 0
        plt.plot(sim.t_vect, eyePos2,color='b')
        tauParams = [np.nan, np.nan, np.nan, np.nan]
        try:
            tauParams = tc.MyCurveFitter(shiftedTime, adjustedEyePos, guess)
            print(tauParams)
        except:
            print("Couldn't find fit")
        A = tauParams[0]
        B = tauParams[1]
        t1 = tauParams[2]
        t2 = tauParams[3]
        print(tauParams)
        if tauParams[0] != np.nan:
            y = [tauParams[0] * np.power(np.e, -t / tauParams[2]) + tauParams[1] * np.power(np.e, -t / tauParams[3]) +
                 eyePos2[-1] for t in sim.t_vect]
            #plt.plot(sim.t_vect, y, color='purple', label="A: " + str(round(A,2)) + " B: " + str(round(B,2)) + " t1: " + str(round(t1,2)) + " t2: " + str(round(t2,2)))
        else:
            y = np.ones(len(sim.t_vect)) * eyePos2[0]
            plt.scatter(sim.t_vect, y, color='purple', label='Exponential Fit')
        pickle.dump((A,B,t1,t2), open("FitParams.bin", "wb"))
        #Normal with half network weakened
        #eyePos3, rVect3, kIdx3 = sim.RunSimFWeakenSide(timeToKill, startIdx=e, weakFrac=.3)
        #shiftedTime = sim.t_vect[kIdx3:]-sim.t_vect[kIdx3] #Sets t=0 to the start of the decay
        #adjustedEyePos = eyePos3[kIdx3:]-eyePos3[-1] #Shifts the eye position down so steady state is 0
        #plt.plot(sim.t_vect, eyePos3, color='r')
        """
        tauParams = [np.nan, np.nan, np.nan, np.nan]
        try:
            tauParams = tc.MyCurveFitter(shiftedTime, adjustedEyePos, guess)
            print(tauParams)
        except:
            continue
       #A[i, eIdx] = tauParams[0]
       # B[i, eIdx] = tauParams[1]
       # t1[i, eIdx] = tauParams[2]
       # t2[i, eIdx] = tauParams[3]
        A = tauParams[0]
        B = tauParams[1]
        t1 = tauParams[2]
        t2 = tauParams[3]

        if tauParams[0] != np.nan:
            y = [tauParams[0]*np.power(np.e, -t/tauParams[2]) + tauParams[1]*np.power(np.e, -t/tauParams[3]) + eyePos3[-1] for t in sim.t_vect]
            plt.plot(sim.t_vect, y, color='purple', label="A: " + str(round(A,2)) + " B: " + str(round(B,2)) + " ts: " + str(round(t1,3)) + " tf: " + str(round(t2,3)))
        else:
            y = np.ones(len(sim.t_vect)) * eyePos3[0]
            plt.scatter(sim.t_vect, y, color='purple', label="A: " + str(round(A,2)) + " B: " + str(round(B,2)) + " ts: " + str(round(t1,3)) + " tf: " + str(round(t2,3)))
        """
        print(eIdx)
        #eIdx = eIdx + 1
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
#xpickle.dump((A,B,t1,t2), open("FitParams.bin", "wb"))

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