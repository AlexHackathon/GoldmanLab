import random
import numpy
import numpy as np
import matplotlib.pyplot as plt
import math
import Helpers.ActivationFunction as ActivationFunction
import Helpers.GraphHelpers as MyGraphing
import Helpers.CurrentGenerator as MyCurrent

class Simulation:
    def __init__(self, neuronNum, dt, end, tau, a, p, maxFreq, eyeStartParam, eyeStopParam, eyeResParam):
        self.neuronNum = neuronNum #Number of neurons in the simulation
        self.dt = dt #Time step [ms]
        self.t_end = end #Simulation end [ms]
        self.tau = tau
        self.t_vect = np.arange(0, self.t_end, self.dt)

        self.eyeStart = eyeStartParam
        self.eyeStop = eyeStopParam
        self.eyeRes = eyeResParam

        self.current_mat = np.zeros((self.neuronNum, len(self.t_vect))) #Defaults to no current
        self.shouldGetCurrent = MyCurrent.ShouldGetBurst(self.t_vect, 100, 300, 200, 1000, .1)
        #self.w_mat = np.zeros((self.neuronNum, self.neuronNum + 1)) #Tonic as last column
        self.w_mat = np.zeros((self.neuronNum,self.neuronNum))
        self.v_mat = np.zeros((len(self.t_vect), neuronNum)) #For simulation

        self.maxFreq = maxFreq
        self.onPoints = np.linspace(self.eyeStart, self.eyeStop, self.neuronNum)
        print(self.onPoints)
        self.onIdx = np.zeros((self.neuronNum))
        self.eyePos = np.linspace(self.eyeStart, self.eyeStop, self.eyeRes)
        self.r_mat = self.CreateTargetCurves()
        self.T = np.zeros((self.neuronNum,))

        self.fixedA = a
        self.fixedP = p
        #Fixed points is a list of the same length as eyePos
        #Each index contains a list of fixed points for that simulation
        #Each fixed point contains the values for each of the neurons at that time point
        self.fixedPoints = [] #NOT A SQUARE MATRIX

    def SetCurrent(self, currentMat):
        self.current_mat = currentMat
    def CreateTargetCurves(self):
        slope = self.maxFreq / (self.eyeStop - 0)
        r_mat = np.zeros((self.neuronNum, len(self.eyePos)))
        for i in range(self.neuronNum):
            switch = False
            for eIdx in range(len(self.eyePos)):
                y = None
                if self.onPoints[i] < 0:
                    #Positive Slope
                    if self.eyePos[eIdx] < self.onPoints[i]:
                        y = 0
                    else:
                        # Point-slope formula y-yi = m(x-xi)
                        y = slope * (self.eyePos[eIdx] - self.onPoints[i])
                        if not switch:
                            switch = True
                            self.onIdx[i] = eIdx
                else:
                    if self.eyePos[eIdx] > self.onPoints[i]:
                        y = 0
                    else:
                        # Point-slope formula y-yi = m(x-xi)
                        y = -slope * (self.eyePos[eIdx] - self.onPoints[i])
                        if not switch:
                            switch = True
                            self.onIdx[i] = eIdx
                r_mat[i][eIdx] = y
        self.r_mat = r_mat

    def PlotTargetCurves(self, rMatParam, eyeVectParam):
        colors = MyGraphing.getDistinctColors(self.neuronNum)
        for r in range(len(rMatParam)):
            # plt.plot(eyeVectParam, rMatParam[r], color = colors[r])
            plt.plot(eyeVectParam, rMatParam[r])
        plt.xlabel("Eye Position")
        plt.ylabel("Firing Rate")
        plt.show()
    def FitWeightMatrixExclude2(self):
        for n in range(self.neuronNum):
            startIdx = int(self.onIdx[n])
            #Do the fit
            X = np.ones((len(self.eyePos)-startIdx, self.neuronNum + 1))
            for i in range(len(X)):
                for j in range(len(X[0])-1):
                    X[i,j] = ActivationFunction.Geometric(self.r_mat[j,i+startIdx], self.fixedA, self.fixedP)
            r = self.r_mat[n][startIdx:]
            solution = np.linalg.lstsq(X,r)[0]
            self.w_mat[n] =solution[:-1]
            self.T[n] = solution[-1]
        self.FitPredictorNonlinear()
    def FitWeightMatrixNew(self):
        #Store firing rate in a matrix of firing rates over eye positions
        #Use scipy.linalg.lsq_linear() to solve for the weight matrix row by row
        #dr/dt and I are 0 because we are only looking at fixed points
        # Setting S_mat (n+1 x e)
        S_mat = ActivationFunction.Geometric(self.r_mat, self.fixedA, self.fixedP)
        #print(np.shape(S_mat,))
        ones = np.array(np.ones(len(S_mat[-1])))
        ones = np.reshape(ones, (1,len(ones)))
        #print(np.shape(ones))
        sTilda = np.append(S_mat, ones, axis=0)
        sTildaTranspose = np.transpose(sTilda)  # Shape: (50,6)
        for k in range(len(self.w_mat)):
            #r_e and S~(r) transpose
            r = np.array(self.r_mat[k]) #Shape: (50,)
            weightSolution = np.linalg.lstsq(sTildaTranspose, r, rcond=None)[0]
            self.w_mat[k] = weightSolution[:-1]
            self.T[k] = weightSolution[-1]
            #self.T[k] = min(0,weightSolution[-1])
        for t in range(len(self.T)):
            self.T[t] = min(self.T[t], 0)
        print(self.T)
        print(self.w_mat)
        self.FitPredictorNonlinear()
    def FitPredictorNonlinear(self):
        S_mat_T = np.zeros((len(self.eyePos),self.neuronNum))
        for i in range(len(S_mat_T)):
            for j in range(len(S_mat_T[0])):
                S_mat_T[i,j] = ActivationFunction.Geometric(self.r_mat[j,i], self.fixedA, self.fixedP)
        print(np.shape(S_mat_T))
        print(np.shape(self.eyePos))
        weightSolution = np.linalg.lstsq(S_mat_T, self.eyePos, rcond=None)[0]
        self.predictW = weightSolution

    def PredictEyePosNonlinear(self, r_E):
        return (np.dot(ActivationFunction.Geometric(r_E, self.fixedA, self.fixedP), self.predictW))
    def RunSim(self, startCond=np.empty(0), plot=False, uneven=False):
        # print("Running sim")
        if not np.array_equal(startCond, np.empty(0)):
            self.v_mat[0] = startCond
        else:
            self.v_mat[0] = sim.r_mat[:, 0]
        tIdx = 1
        eyePositions = []
        while tIdx < len(self.t_vect):
            # Sets the basic values of the frame
            v = self.v_mat[tIdx - 1]
            activation = ActivationFunction.Geometric(v, self.fixedA, self.fixedP)
            dot = np.dot(self.w_mat, activation)
            if not uneven:
                delta = (-v + dot + self.T + self.current_mat[:, tIdx])
            else:
                #Find Should Get Bursts
                newCurrent = v / np.linalg.norm(v) * 2
                delta = (-v + dot + self.T + newCurrent)
            self.v_mat[tIdx] = self.v_mat[tIdx - 1] + self.dt / self.tau * delta
            for i in range(len(self.v_mat[tIdx])):
                self.v_mat[tIdx][i] = max(0, self.v_mat[tIdx][i])
            if plot:
                if tIdx == 1:
                    # Double adds the first eye position to correct for starting at 1
                    eyePositions.append(self.PredictEyePosNonlinear(self.v_mat[tIdx]))
                    eyePositions.append(self.PredictEyePosNonlinear(self.v_mat[tIdx]))
                else:
                    eyePositions.append(self.PredictEyePosNonlinear(self.v_mat[tIdx]))
            tIdx += 1
        if plot:
            # Calculates the eye position at each time point in the simulation
            plt.plot(self.t_vect, eyePositions)
            # plt.show()
        plt.xlabel("Time (ms)")
        plt.ylabel("Eye Position (degrees)")

neurons = 300
#(self, neuronNum, dt, end, tau, a, p, maxFreq, eyeStartParam, eyeStopParam, eyeResParam):
sim = Simulation(neurons, .1, 1000, 20, .4, 1.4, 75, -50, 50, 300)
sim.CreateTargetCurves()
sim.PlotTargetCurves(sim.r_mat,sim.eyePos)
#sim.FitWeightMatrixNew()
sim.FitWeightMatrixExclude2()
for e in range(len(sim.eyePos)):
    if e%50 == 0:
        print(e)
        sim.RunSim(startCond=sim.r_mat[:,e], plot=True, uneven=False)
plt.show()
"""
sim.SetCurrent(MyCurrent.ConstCurrentBursts(sim.t_vect, 2, 100, 300, 0, 6000, neurons, sim.dt))
for e in range(len(sim.eyePos)):
    if e%50 == 0:
        print(e)
        sim.RunSim(startCond=sim.r_mat[:,e], plot=True)
plt.show()"""
"""sim.SetCurrent(MyCurrent.ConstCurrentBursts(sim.t_vect, -2, 100, 300, 0, 6000, neurons, sim.dt))
for e in range(len(sim.eyePos)):
    if e%50 == 0:
        print(e)
        sim.RunSim(startCond=sim.r_mat[:,e], plot=True)
plt.show()"""
#DOESN"T WORK BECAUSE THE CURRENT ISN'T BEING APPLIED ALONG THE INTEGRATING MODE (+/- THE CURRENT FIRING RATES)