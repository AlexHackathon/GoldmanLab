import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize
import multiprocessing

import Helpers.ActivationFunction as ActivationFunction
import Helpers.CurrentGenerator
import TuningCurves
import Helpers.Bound
import TestTauCalculator as tc

import pandas as pd
import pickle
import os

#r_star = 1
#n_Ca = 4
class Simulation:
    def __init__(self, simulationParameters, fileLocation):
        #self.neuronNum = neuronNum #Number of neurons in the simulation
        self.dt = simulationParameters.dt #Time step [ms]
        self.t_end = simulationParameters.end #Simulation end [ms]
        self.t_vect = np.arange(0, self.t_end, self.dt) #A time vector ranging from 0 to self.t_end

        self.eyeStart = simulationParameters.eyeStartParam #Start of eye positions (degrees)
        self.eyeStop = simulationParameters.eyeStopParam #End of eye positions (degrees)
        self.eyeRes = simulationParameters.eyeResParam #Number of points between the start and end
        self.maxFreq = simulationParameters.maxFreq #Highest frequency reached by a neuron

        self.tau = simulationParameters.tau #Time constant
        self.f_noR = simulationParameters.nonlinearityNoR #The nonlinearity used in the network not multiplied by firing rate
        self.f = simulationParameters.nonlinearity #The sole nonlinearity used in this network

        self.eyePos = np.linspace(self.eyeStart, self.eyeStop, self.eyeRes) #Vector of eye positions
        self.r_mat = TuningCurves.TwoSidedDifferentSlopeMirror(fileLocation, self.eyeStart, self.eyeStop, self.eyeRes)
        self.r_mat_neg = TuningCurves.TwoSidedDifferentSlopeMirrorNeg(fileLocation, self.eyeStart, self.eyeStop, self.eyeRes)
        self.cutoffIdx = TuningCurves.GetCutoffIdxMirror(fileLocation, self.eyeStart, self.eyeStop, self.eyeRes)
        self.r_mat = np.array(self.r_mat)
        self.neuronNum = len(self.r_mat)
        self.w_mat = np.zeros((self.neuronNum,self.neuronNum)) #nxn matrix of weights
        self.current_mat = np.zeros((self.neuronNum, len(self.t_vect)))  # Defaults to no current
        self.predictW = None  # Weight vector used to predict eye position from firing rates
        self.predictT = None  # Tonic input for adjustments to the predicted eye position
        self.s_mat = np.zeros((len(self.t_vect), self.neuronNum))  # txn 2d array for storing information from the simulations
        self.T = np.zeros((self.neuronNum,))  # Tonic input to all the neurons

        #Facilitation Threshold Variables
        self.f_f = simulationParameters.f
        self.t_f = simulationParameters.t_f
        self.P0_f = simulationParameters.P0

        #Mark edge neurons
        self.minIdx = TuningCurves.FindMin(fileLocation)
    def ReadWeightMatrix(self, wfl, ewfl,Tfl):
        readMatrix=pickle.load(open(wfl,"rb"))
        self.w_mat = readMatrix
        readMatrixT = pickle.load(open(Tfl,"rb"))
        self.T = readMatrixT
        readMatrixEye = pickle.load(open(ewfl,"rb"))
        self.predictW = readMatrixEye[:-1]
        self.predictT = readMatrixEye[-1]

    def WriteWeightMatrix(self, matrix, fileName):
        print(np.shape(matrix))
        pickle.dump(matrix, open(fileName, "wb"))
    def FitPredictorNonlinearSaturation(self):
        '''Fit the weight vector for predicting eye position.

        Create a matrix of the activation function of firing rates.
        Fit weight and a constant to predict eye position from
        firing rates passed through an activation function.'''
        S_mat_T = np.ones((len(self.eyePos),self.neuronNum+1))
        for i in range(len(S_mat_T)):
            for j in range(len(S_mat_T[0])-1):
                S_mat_T[i,j] = self.f(self.r_mat[j,i])
        #CHANGE: potentially include a tonic input in the prediction
        weightSolution = np.linalg.lstsq(S_mat_T, self.eyePos, rcond=None)[0]
        self.predictW = weightSolution[:-1]
        self.predictT = weightSolution[-1]

    def PredictEyePosNonlinearSaturation(self, s_E):
        '''Predict eye positions.

        Parameters
        s_E: a vector of activation at a given time point

        Returns predicted eye positions (constant).'''
        x = np.dot(s_E, self.predictW)
        y = x + self.predictT
        return y
    def GetR(self, s_e):
        '''Predict firing rates.

        Parameters
        s_E: a vector of activation at a given time point

        Returns
        r_E: a vector of firing rates at a given time point'''
        r_e = np.dot(self.w_mat, s_e) + self.T
        return r_e
    def RunSimF(self, timeAtKill, startIdx=-1, dead=[]):
        '''Run simulation generating activation values. (Facilitation)

        Set the starting value to the activation function of the target firing rates.
        Update using the update rule: t * ds/dt = -s + P_rel * r.

        P_rel*r is wrapped in self.f()'''
        if not startIdx == -1:
            self.s_mat[0,:] = self.f(self.r_mat[:, startIdx])
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0,:] = self.f(self.r_mat[:, startIdx])
        #Set default values
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0,:])
        Rs = np.zeros((len(self.t_vect), self.neuronNum)) #Could be changed to just keep last to make more efficient
        #If behavior is expected, we would also know all the firing rates from the tuning curves (CHECK)
        P0_vect = np.ones((self.neuronNum)) * self.P0_f
        P_rel = np.zeros((len(self.t_vect), self.neuronNum))
        P_rel[0] = np.array(self.f_noR(self.r_mat[:,startIdx]))
        growthMat = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            #Remove the firing rate of the dead neurons
            if self.t_vect[tIdx] > timeAtKill:
                for d in dead:
                    r_vect[d] = 0 #DONT NEGATE THE OTHER TWO VARIABLES
            Rs[tIdx]=r_vect
            changeP = -P_rel[tIdx-1] + P0_vect + self.t_f * self.f_f*np.multiply(r_vect, (1-P_rel[tIdx-1])) #Maybe change the weights to be large
            P_rel[tIdx] = P_rel[tIdx-1] + self.dt/self.t_f * changeP
            decay = -self.s_mat[tIdx - 1]
            growth = np.multiply(P_rel[tIdx-1], r_vect)
            growthMat[tIdx] = growth
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            #Predict eye position based on synaptic activation
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            #Increment the time index
            tIdx += 1
        return eyePositions, Rs, tc.CalculateTau(self.t_vect, eyePositions)
#External Functions
def GetDeadNeurons(fraction, firstHalf, neuronNum):
    if firstHalf:
        return range(0, int(neuronNum // 2 * fraction))
    else:
        return [neuronNum // 2 + j for j in range(0, int(neuronNum // 2 * fraction))]
def FitHelperParallel(startN, endN, X, bounds, r_mat_neg, sim):
    for myN in range(startN,endN):
        r = r_mat_neg[myN]
        solution = None
        if bounds != None:
            bounds = bounds[myN]
            solution = scipy.optimize.lsq_linear(X, r, bounds)
        else:
            solution = scipy.optimize.lsq_linear(X, r)
        sim.w_mat[myN] = solution.x[:-1]
        sim.T[myN] = solution.x[-1]
def FitWeightMatrixExcludeParallel(fileLoc, eyeFileLoc, tFileLoc, sim, bounds):
    '''Fit fixed points in the network using target curves.

    Create an activation function matrix X (exn+1).
    Fit each row of the weight matrix with linear regression.
    Call the function to fit the predictor of eye position.
    Exclude eye positions where a neuron is at 0 for training each row.'''
    #POTENTIAL CHANGE: Take ten rows and execute them on a core
    #Update the program to say that those have been taken
    #When the other core finishes, it updates which it has taken
    #If the last has been taken, continue to writing
    X = np.ones((len(sim.eyePos), len(sim.r_mat) + 1))
    for i in range(len(X)):
        for j in range(len(X[0]) - 1):
            X[i, j] = sim.f(sim.r_mat[j, i])
    p1 = multiprocessing.Process(target=FitHelperParallel, args=(0, sim.neuronNum//2, X, bounds, sim.r_mat_neg, sim))
    p2 = multiprocessing.Process(target=FitHelperParallel, args=(sim.neuronNum//2, sim.neuronNum, X, bounds, sim.r_mat_neg, sim))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    sim.WriteWeightMatrix(sim.w_mat, fileLoc)
    sim.WriteWeightMatrix(sim.T, tFileLoc)
    sim.FitPredictorNonlinearSaturation(eyeFileLoc)

class FacilitationParameters:
    def __init__(self, dt, end, tau, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, P0, f, t_f):
        self.dt = dt
        self.end = end
        self.tau = tau
        self.maxFreq = maxFreq
        self.eyeStartParam = eyeStartParam
        self.eyeStopParam = eyeStopParam
        self.eyeResParam = eyeResParam
        self.P0 = P0
        self.f = f
        self.t_f = t_f
        self.nonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, P0, f, t_f)
        self.nonlinearityNoR = lambda r_vect: ActivationFunction.SynapticFacilitationNoR(r_vect, P0, f, t_f)


