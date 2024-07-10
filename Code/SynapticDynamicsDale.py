import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg

import Helpers.ActivationFunction as ActivationFunction
import scipy as sp

class Simulation:
    def __init__(self, neuronNum, dt, end, tau, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, nonLinearityFunction):
        self.neuronNum = neuronNum #Number of neurons in the simulation
        self.dt = dt #Time step [ms]
        self.t_end = end #Simulation end [ms]
        self.t_vect = np.arange(0, self.t_end, self.dt)

        self.eyeStart = eyeStartParam
        self.eyeStop = eyeStopParam
        self.eyeRes = eyeResParam
        self.maxFreq = maxFreq

        self.tau = tau
        self.f = nonLinearityFunction

        self.eida = np.zeros((self.neuronNum, self.neuronNum))

        self.onPoints = np.append(np.linspace(self.eyeStart, overlap, self.neuronNum//2), np.linspace(-overlap, self.eyeStop, self.neuronNum//2))
        self.cutoffIdx = np.zeros((self.neuronNum))
        self.eyePos = np.linspace(self.eyeStart, self.eyeStop, self.eyeRes)

        self.current_mat = np.zeros((self.neuronNum, len(self.t_vect))) #Defaults to no current
        self.w_mat = np.zeros((self.neuronNum,self.neuronNum))
        self.predictW = None
        self.predictT = None
        self.s_mat = np.zeros((len(self.t_vect), self.neuronNum))
        self.T = np.zeros((self.neuronNum,))
        self.r_mat = self.CreateTargetCurves()

    def SetCurrent(self, currentMat):
        self.current_mat = currentMat
    def SetWeightMatrix(self, weightMatrix):
        self.w_mat = weightMatrix
    def CreateTargetCurves(self):
        slope = self.maxFreq / (self.eyeStop - self.eyeStart)
        r_mat = np.zeros((self.neuronNum, len(self.eyePos)))
        for n in range(self.neuronNum):
            switch = False
            for eIdx in range(len(self.eyePos)):
                #If neurons have positive slope and have 0s at the start
                y = None
                if n < self.neuronNum//2:
                    y = slope * (self.eyePos[eIdx] - self.onPoints[n])
                    # Correction for negative numbers
                    if y < 0:
                        y = 0
                    else:
                        if not switch:
                            # Marks where the first positive number occurs
                            switch = True
                            self.cutoffIdx[n] = eIdx
                #If neurons have negative slope and end with 0s
                else:
                    y = -slope * (self.eyePos[eIdx] - self.onPoints[n])
                    if y < 0:
                        y = 0
                        if not switch:
                            # Marks where the first negative number occurs
                            switch = True
                            self.cutoffIdx[n] = eIdx
                r_mat[n][eIdx] = y
            #If it is positive for all eye positions or 0 for all eye positions
            if not switch:
                self.cutoffIdx[n] = len(self.eyePos)-1
        self.r_mat = r_mat
    def PlotTargetCurves(self):
        for r in range(len(self.r_mat)):
            plt.plot(self.eyePos, self.r_mat[r])
        plt.xlabel("Eye Position")
        plt.ylabel("Firing Rate")
        plt.show()
    def FitWeightMatrixExcludeBilateralSaturating(self):
        #X = np.ones((len(self.eyePos), self.neuronNum + 1))
        #for i in range(len(X)):
        #    for j in range(len(X[0]) - 1):
        #        X[i, j] = self.f(self.r_mat[j, i])
        X = np.ones((self.neuronNum+1, len(self.eyePos)))
        X[:-1,:] = self.f(self.r_mat)
        for n in range(self.neuronNum):
            #Set the bounds to be excitatory same side
            #and inhibitory to the opposite side.
            #bounds = self.BoundQuadrants(n)
            bounds = self.BoundDale(n)
            #Run the fit with the specified bounds
            guess = np.zeros((self.neuronNum + 1))
            func = lambda w_n: self.RFitFunc(w_n, X, self.r_mat[n])
            solution = sp.optimize.minimize(func, guess,bounds=bounds)
            print(n)
            self.w_mat[n] = solution.x[:-1]
            self.T[n] = solution.x[-1]
        self.FitPredictorNonlinearSaturation()
    def BoundDale(self,n):
        bounds = [0 for n in range(self.neuronNum + 1)]
        bounds[-1] = (None, None)
        """for nIdx in range(self.neuronNum):
            if nIdx < self.neuronNum // 4:
                bounds[nIdx] = (0, None)
            elif nIdx < self.neuronNum // 2:
                bounds[nIdx] = (None, 0)
            elif nIdx < 3 * self.neuronNum // 4:
                bounds[nIdx] = (0, None)
            else:
                bounds[nIdx] = (None, 0)"""
        if n < self.neuronNum // 2:
            for nIdx in range(self.neuronNum):
                if nIdx < self.neuronNum // 4:
                    bounds[nIdx] = (0, None)
                elif nIdx < self.neuronNum // 2:
                    bounds[nIdx] = (0,0)
                elif nIdx < 3*self.neuronNum//4:
                    bounds[nIdx] = (None, 0)
                else:
                    bounds[nIdx] = (0,0)
        else:
            for nIdx in range(self.neuronNum):
                if nIdx < self.neuronNum // 4:
                    bounds[nIdx] = (0, 0)
                elif nIdx < self.neuronNum // 2:
                    bounds[nIdx] = (None, 0)
                elif nIdx < 3 * self.neuronNum // 4:
                    bounds[nIdx] = (0, 0)
                else:
                    bounds[nIdx] = (0, None)
        return bounds
    def BoundQuadrants(self, n):
        bounds = [0 for n in range(self.neuronNum + 1)]
        bounds[-1] = (None, None)
        if n < self.neuronNum // 2:
            for nIdx in range(self.neuronNum):
                if nIdx < self.neuronNum // 2:
                    bounds[nIdx] = (0, None)
                else:
                    bounds[nIdx] = (None, 0)
        else:
            for nIdx in range(self.neuronNum):
                if nIdx < self.neuronNum // 2:
                    bounds[nIdx] = (None, 0)
                else:
                    bounds[nIdx] = (0, None)
        return bounds
    def FitPredictorNonlinearSaturation(self):
        S_mat_T = np.ones((len(self.eyePos),self.neuronNum+1))
        for i in range(len(S_mat_T)):
            for j in range(len(S_mat_T[0])-1):
                S_mat_T[i,j] = self.f(self.r_mat[j,i])
        #CHANGE: potentially include a tonic input in the prediction
        weightSolution = np.linalg.lstsq(S_mat_T, self.eyePos, rcond=None)[0]
        self.predictW = weightSolution[:-1]
        self.predictT = weightSolution[-1]

    def PredictEyePosNonlinearSaturation(self, s_E):
        #Change: potentially include a tonic input in the prediction
        return np.dot(s_E, self.predictW) + self.predictT
    def GetR(self, s_e):
        r_e = np.dot(self.w_mat, s_e) + self.T
        return r_e
    def RFitFunc(self, w_n, S, r):
        #x is w_i* with tonic attached to the end
        #y is s_e with extra 1 at the end
        #S must be nxe
        return abs(np.linalg.norm(np.dot(w_n, S) - r))
    def GraphWeightMatrix(self):
        plt.imshow(self.w_mat)
        plt.title("Weight Matrix")
        plt.colorbar()
        plt.show()
    def MistuneMatrix(self, fractionOffset = .01):
        self.w_mat = (1-fractionOffset) * self.w_mat
    def RunSim(self, startIdx=-1, plot=True, dead=[]):
        if not startIdx == -1:
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0])
        while tIdx < len(self.t_vect):
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T)
            r_vect = [0 if r < 0 else r for r in r_vect] #CHANGE
            for d in dead:
                r_vect[d] = 0
            decay = -self.s_mat[tIdx - 1]
            growth = self.f(r_vect)
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            tIdx += 1
        if plot:
            plt.plot(self.t_vect, eyePositions)
        plt.xlabel("Time (ms)")
        plt.ylabel("Eye Position (degrees)")
    def RunSimTau(self, startIdx=-1, plot=True, dead=[]):
        if not startIdx == -1:
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0])
        while tIdx < len(self.t_vect):
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T)
            for r in range(len(r_vect)):
                if r_vect[r] < 0:
                    r_vect[r] = 0
            for d in dead:
                r_vect[d] = 0
            decay = -self.s_mat[tIdx - 1]
            growth = alpha * r_vect
            for g in range(len(growth)):
                growth[g] = (1-self.s_mat[tIdx-1][g]) * growth[g]
            growth = np.array(growth)
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            tIdx += 1
        if plot:
            plt.plot(self.t_vect, eyePositions)
        plt.xlabel("Time (ms)")
        plt.ylabel("Eye Position (degrees)")
    def PlotFixedPointsOverEyePos(self,neuronArray):
        y = np.zeros((len(self.eyePos),self.neuronNum))
        s = np.zeros((len(self.eyePos),self.neuronNum))
        for e in range(len(self.eyePos)):
            s_vect = self.f(self.r_mat[:,e])
            r_calc = self.GetR(s_vect)
            for r in range(len(r_calc)):
                if r_calc[r] < 0:
                    r_calc[r] = 0
            growth = self.f(r_calc)
            y[e] = growth
            s[e] = s_vect
        for idx in neuronArray:
            plt.plot(self.eyePos, y[:,idx], label = "growth")
            plt.plot(self.eyePos, s[:,idx], label = "decay")
        plt.legend()
        plt.xlabel("Eye Position")
        plt.ylabel("Fixed Points")
    def PlotFixedPointsOverEyePosRate(self,neuronArray):
        for n in neuronArray:
            r = np.zeros(len(self.eyePos))
            for e in range(len(self.eyePos)):
                r[e] = np.dot(self.w_mat[n], self.f(self.r_mat[:,e])) + self.T[n]
                r[e] = max(0, r[e])
            plt.plot(self.eyePos, self.r_mat[n], label = "decay")
            plt.plot(self.eyePos, r, label = "growth")
            plt.legend()
            plt.xlabel("Eye Position")
            plt.ylabel("Fixed Points")

overlap = 5
neurons = 150
#(self, neuronNum, dt, end, tau, a, p, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, nonlinearityFunction):
#Instantiate the simulation
alpha = .05 #1 works. .05 works even better for synaptic nonlinearity
myNonlinearity = lambda r_vect: ActivationFunction.SynapticSaturation(r_vect, alpha)
#myNonlinearity = lambda r_vect: alpha * ActivationFunction.Geometric(r_vect, .4, 1.4)
sim = Simulation(neurons, .01, 100, 100, 150, -25, 25, 2000, myNonlinearity)
#Create and plot the curves
sim.CreateTargetCurves()
#sim.PlotTargetCurves(sim.r_mat,sim.eyePos)
#sim.FitWeightMatrix()
sim.FitWeightMatrixExcludeBilateralSaturating()
sim.FitPredictorNonlinearSaturation()
#sim.PlotFixedPointsOverEyePosRate([0])
#plt.show()

#Reverse Engineer Target Curves
"""M = np.zeros((len(sim.eyePos), sim.neuronNum))
for e in range(len(sim.eyePos)):
    s = sim.f(sim.r_mat[:,e])
    r = np.dot(sim.w_mat, s) + sim.T
    M[e] = r - sim.r_mat[:,e]
    plt.scatter(np.ones(len(r))*sim.eyePos[e], r)
plt.show()

plt.imshow(M)
plt.colorbar()
plt.show()"""
"""for e in range(len(sim.eyePos)):
    predict = sim.PredictEyePosNonlinearSaturation(sim.f(sim.r_mat[:,e]))
    plt.scatter(sim.eyePos[e], predict)
plt.show()"""

for e in range(len(sim.eyePos)):
    if e%100 == 0:
        sim.RunSimTau(e, plot=True)
plt.show()

#Graph Weight Matrix
plt.imshow(sim.w_mat,cmap="seismic",norm=matplotlib.colors.CenteredNorm())
plt.colorbar()
plt.show()