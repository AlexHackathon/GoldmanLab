import numpy as np
import matplotlib.pyplot as plt
import Helpers.ActivationFunction as ActivationFunction
import Helpers.GraphHelpers as MyGraphing
import Helpers.CurrentGenerator as MyCurrent

overlap = 5
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
        self.w_mat = np.zeros((self.neuronNum,self.neuronNum))
        self.v_mat = np.zeros((len(self.t_vect), self.neuronNum)) #For simulation
        self.ksi = np.zeros((self.neuronNum,))

        self.maxFreq = maxFreq
        self.onPoints = np.append(np.linspace(self.eyeStart, overlap, self.neuronNum//2), np.linspace(-overlap, self.eyeStop, self.neuronNum//2))
        self.cutoffIdx = np.zeros((self.neuronNum))
        self.eyePos = np.linspace(self.eyeStart, self.eyeStop, self.eyeRes)
        self.T = np.zeros((self.neuronNum,))
        self.r_mat = self.CreateTargetCurves()

        self.fixedA = a
        self.fixedP = p

    def SetCurrent(self, currentMat):
        self.current_mat = currentMat
    def CreateTargetCurves(self):
        slope = self.maxFreq / (self.eyeStop - self.eyeStart)
        r_mat = np.zeros((self.neuronNum, len(self.eyePos)))
        for n in range(self.neuronNum):
            switch = False
            for eIdx in range(len(self.eyePos)):
                #If neurons have positive slope and have 0s at the start
                y = None
                if n < self.neuronNum//2:
                    self.ksi[n] = slope
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
                    self.ksi[n] = -slope
                    y = -slope * (self.eyePos[eIdx] - self.onPoints[n])
                    if y < 0:
                        y = 0
                        if not switch:
                            # Marks where the first positive number occurs
                            switch = True
                            self.cutoffIdx[n] = eIdx
                r_mat[n][eIdx] = y
            if not switch:
                self.cutoffIdx[n] = len(self.eyePos)-1
            #Set the tonic input to the non-corrected y intercept
            if n < self.neuronNum // 2:
                self.T[n] = slope * (self.eyeStart - self.onPoints[n]) #Intercept with x=-25
            else:
                self.T[n] = -slope * (self.eyeStop - self.onPoints[n]) #Intercept with x=25
        self.r_mat = r_mat

    def PlotTargetCurves(self, rMatParam, eyeVectParam):
        colors = MyGraphing.getDistinctColors(self.neuronNum)
        for r in range(len(rMatParam)):
            # plt.plot(eyeVectParam, rMatParam[r], color = colors[r])
            plt.plot(eyeVectParam, rMatParam[r])
        plt.xlabel("Eye Position")
        plt.ylabel("Firing Rate")
        plt.show()
    def FitWeightMatrixExcludeBilateral(self):
        X = np.ones((len(self.eyePos), self.neuronNum + 1))
        for i in range(len(X)):
            for j in range(len(X[0]) - 1):
                X[i, j] = ActivationFunction.Geometric(self.r_mat[j, i], self.fixedA, self.fixedP)
        for n in range(self.neuronNum):
            startIdx = int(self.cutoffIdx[n])
            # Do the fit
            # Two different because the two sides will want different sides of the matrix
            if n < self.neuronNum // 2:
                r = self.r_mat[n][startIdx:]
                solution = np.linalg.lstsq(X[startIdx:, :], r)[0]
                self.w_mat[n] = solution[:-1]
                self.T[n] = solution[-1]
            else:
                r = self.r_mat[n][:startIdx]
                solution = np.linalg.lstsq(X[:startIdx, :], r)[0]
                self.w_mat[n] = solution[:-1]
                self.T[n] = solution[-1]
        print(self.T[self.neuronNum // 2:])
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
    def GraphWeightMatrix(self):
        plt.imshow(self.w_mat)
        plt.title("Weight Matrix")
        plt.colorbar()
        plt.show()
    def RunSim(self, startIdx=-1, plot=False):
        # print("Running sim")
        if not startIdx == -1:
            self.v_mat[0] = sim.r_mat[:, startIdx]
        else:
            #Set it to some default value in the middle
            self.v_mat[0] = sim.r_mat[:, self.neuronNum//2]
            startIdx = self.neuronNum//2
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        print(startIdx)
        print(len(self.eyePos))
        print(np.shape(self.eyePos))
        eyePositions[0] = self.eyePos[startIdx]
        s_mat_sim = []
        #Simulation fills up v_mat, eyePositions, s_mat_sim
        while tIdx < len(self.t_vect):
            # Sets the basic values of the frame
            funcIn = self.ksi * eyePositions[tIdx-1] + self.T #nx1 vector
            dot = np.dot(self.eta, ActivationFunction.Geometric(funcIn, self.fixedA, self.fixedP))
            delta = (-eyePositions[tIdx-1] + dot)
            eyePositions[tIdx] = eyePositions[tIdx - 1] + self.dt / self.tau * delta
            tIdx += 1
        if plot:
            # Calculates the eye position at each time point in the simulation
            plt.plot(self.t_vect, eyePositions)
            # plt.show()
        plt.xlabel("Time (ms)")
        plt.ylabel("Eye Position (degrees)")

neurons = 800
#(self, neuronNum, dt, end, tau, a, p, maxFreq, eyeStartParam, eyeStopParam, eyeResParam):
#Instantiate the simulation
sim = Simulation(neurons, .1, 1000, 20, .4, 1, 150, -25, 25, 600)
#Create and plot the curves
sim.CreateTargetCurves()
sim.PlotTargetCurves(sim.r_mat,sim.eyePos)
sim.FitWeightMatrixExcludeBilateral()

for e in range(len(sim.eyePos)):
    if e%50 == 0:
        sim.RunSim(startIdx=e, plot=True)
plt.show()