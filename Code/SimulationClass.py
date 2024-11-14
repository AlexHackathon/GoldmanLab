import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize

import Helpers.ActivationFunction as ActivationFunction
import Helpers.CurrentGenerator
import TuningCurves
import Helpers.Bound

import pandas as pd
import os

#r_star = 1
#n_Ca = 4
#weightFileLoc = "NeuronWeights.csv"
#eyeWeightFileLoc = "EyeWeights.csv"
class Simulation:
    def __init__(self, neuronNum, dt, end, tau, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, nonLinearityFunction, fileLocation):
        #self.neuronNum = neuronNum #Number of neurons in the simulation
        self.dt = dt #Time step [ms]
        self.t_end = end #Simulation end [ms]
        self.t_vect = np.arange(0, self.t_end, self.dt) #A time vector ranging from 0 to self.t_end

        self.eyeStart = eyeStartParam #Start of eye positions (degrees)
        self.eyeStop = eyeStopParam #End of eye positions (degrees)
        self.eyeRes = eyeResParam #Number of points between the start and end
        self.maxFreq = maxFreq #Highest frequency reached by a neuron

        self.tau = tau #Time constant
        self.f = nonLinearityFunction #The sole nonlinearity used in this network

        #Marks what x-intercept each neuron becomes positive or negative at (depending on side)
        #self.onPoints = np.append(np.linspace(self.eyeStart, overlap, self.neuronNum//2), np.linspace(-overlap, self.eyeStop, self.neuronNum//2))
        #Vector that marks where each neuron becomes positive
        self.eyePos = np.linspace(self.eyeStart, self.eyeStop, self.eyeRes) #Vector of eye positions


        #self.r_mat =  np.zeros((self.neuronNum, len(self.eyePos)))#Tuning curves created for training the network's fixed points
        #self.r_mat_neg = np.zeros((self.neuronNum, len(self.eyePos)))  # Tuning curves negative created for training the network's fixed points
        #self.CreateTargetCurves()
        #self.CreateTargetCurvesNeg()
        #self.CreateAsymmetricalTargetCurves("/Users/alex/Downloads/EmreThresholdSlope_NatNeuroCells_All.xls")
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

        #NEW (Facilitation Threshold Variables)
        self.n_Ca = 1
        self.r_Star = 1
        self.f_f = .4
        self.t_f = 500
        self.P0_f = .1
        self.r0_f = 50
    def SetFacilitationValues(self, n, rStar, f, t_f, P0, r0):
        self.n_Ca = n
        self.r_Star = rStar
        self.f_f = f
        self.t_f = t_f
        self.PO_f = P0
        self.r0_f = r0

    def SetCurrentSplit(self, currentVect):
        '''Sets the current currentVect for first n/2 neurons and -currentVect for last n/2 neurons'''
        for n in range(self.neuronNum):
            if n < self.neuronNum//2:
                self.current_mat[n] = currentVect
            else:
                self.current_mat[n] = -currentVect
    def SetCurrentDoubleSplit(self, currentVect):
        '''Sets the current currentVect for first n/2 neurons and -currentVect for last n/2 neurons'''
        for n in range(self.neuronNum):
            if n < self.neuronNum//2:
                self.current_mat[n] = currentVect
            else:
                self.current_mat[n] = -currentVect
        self.current_mat[:,len(self.t_vect)//2:] = -self.current_mat[:,len(self.t_vect)//2:]
    def SetWeightMatrix(self, weightMatrix):
        '''Sets the weight matrix to the given matrix (nxn).'''
        self.w_mat = weightMatrix
    def SetWeightMatrixRead(self, wfl, ewfl):
        df = pd.read_csv(wfl)
        self.w_mat = df.values[:-1]
        self.T = df.values[-1]
        df2 = pd.read_csv(ewfl)
        self.predictW = df2.values[:-1]
        self.predictT = df2.values[-1]
    def WriteWeightMatrix(self, matrix, fileName):
        #Create a DataFrame from the matrix
        df = pd.DataFrame(matrix)
        #Delete any old file
        try:
            os.remove(fileName)
        except:
            print("No file to override")
        #Write the DataFrame to an csv file
        df.to_csv(fileName)
    #MOVE FUNC TO TUNING CURVES
    def CreateTargetCurves(self):
        '''Create target tuning curves.

        Calculate slope given self.eyeStart, self.eyeStop, self.maxFreq.
        Create the line for a neuron based on the x intercept given by self.onPoints.
        Mark indicies where a neuron begins to be non-zero or begins to be zero (based on side).'''
        slope = self.maxFreq / (self.eyeStop - self.eyeStart)
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
                self.r_mat[n][eIdx] = y
            #If it is positive for all eye positions or 0 for all eye positions
            if not switch:
                self.cutoffIdx[n] = len(self.eyePos)-1
    #MOVE FUNC TO TUNING CURVES
    def CreateTargetCurvesNeg(self):
        '''Create target tuning curves.

        Calculate slope given self.eyeStart, self.eyeStop, self.maxFreq.
        Create the line for a neuron based on the x intercept given by self.onPoints.
        Does not correct for negative values.'''
        slope = self.maxFreq / (self.eyeStop - self.eyeStart)
        for n in range(self.neuronNum):
            for eIdx in range(len(self.eyePos)):
                #If neurons have positive slope
                if n < self.neuronNum // 2:
                    y = slope * (self.eyePos[eIdx] - self.onPoints[n])
                # If neurons have negative slope
                else:
                    y = -slope * (self.eyePos[eIdx] - self.onPoints[n])
                self.r_mat_neg[n][eIdx] = y

    # MOVE FUNC TO TUNING CURVES
    def PlotTargetCurves(self):
        '''Plot given target curves over eye position.'''
        for r in range(len(self.r_mat)):
            plt.plot(self.eyePos, self.r_mat[r])
        plt.xlabel("Eye Position")
        plt.ylabel("Firing Rate")
        plt.show()
    #MOVE FUNC TO BOUND
    def BoundDale(self,n):
        '''Return a vector of restrictions based on Dale's law.

        Prevent a column for having more than one sign. (This represents
        how a neuron can only act as an excitatory or inhibitory neuron)'''
        upperBounds = [0 for n in range(self.neuronNum+1)]
        lowerBounds = [0 for n in range(self.neuronNum+1)]
        upperBounds[-1] = np.inf
        lowerBounds[-1] = -np.inf
        zero = .000000000001
        if n < self.neuronNum // 4:
            for nIdx in range(self.neuronNum):
                if nIdx < self.neuronNum // 4:
                    upperBounds[nIdx] = np.inf
                    lowerBounds[nIdx] = 0
                elif nIdx < self.neuronNum // 2:
                    upperBounds[nIdx] = zero
                    lowerBounds[nIdx] = -zero
                elif nIdx < 3 * self.neuronNum // 4:
                    upperBounds[nIdx] = zero
                    lowerBounds[nIdx] = -zero
                else:
                    upperBounds[nIdx] = 0
                    lowerBounds[nIdx] = -np.inf
        elif n < self.neuronNum // 2:
            for nIdx in range(self.neuronNum):
                if nIdx < self.neuronNum // 4:
                    upperBounds[nIdx] = np.inf
                    lowerBounds[nIdx] = 0
                elif nIdx < self.neuronNum // 2:
                    upperBounds[nIdx] = zero
                    lowerBounds[nIdx] = -zero
                elif nIdx < 3 * self.neuronNum // 4:
                    upperBounds[nIdx] = zero
                    lowerBounds[nIdx] = -zero
                else:
                    upperBounds[nIdx] = 0
                    lowerBounds[nIdx] = -np.inf
        elif n < 3 * self.neuronNum // 4:
            for nIdx in range(self.neuronNum):
                if nIdx < self.neuronNum // 4:
                    upperBounds[nIdx] = zero
                    lowerBounds[nIdx] = -zero
                elif nIdx < self.neuronNum // 2:
                    upperBounds[nIdx] = 0
                    lowerBounds[nIdx] = -np.inf
                elif nIdx < 3 * self.neuronNum // 4:
                    upperBounds[nIdx] = np.inf
                    lowerBounds[nIdx] = 0
                else:
                    upperBounds[nIdx] = zero
                    lowerBounds[nIdx] = -zero
        else:
            for nIdx in range(self.neuronNum):
                if nIdx < self.neuronNum // 4:
                    upperBounds[nIdx] = zero
                    lowerBounds[nIdx] = -zero
                elif nIdx < self.neuronNum // 2:
                    upperBounds[nIdx] = 0
                    lowerBounds[nIdx] = -np.inf
                elif nIdx < 3 * self.neuronNum // 4:
                    upperBounds[nIdx] = np.inf
                    lowerBounds[nIdx] = 0
                else:
                    upperBounds[nIdx] = zero
                    lowerBounds[nIdx] = -zero
        return (lowerBounds, upperBounds)
    #MOVE FUNC TO BOUND
    def BoundQuadrants(self, n,wMax, wMin):
        '''Return a vector of restrictions based on same side excitation
        and opposite side inhibition.'''
        newNeuronNum = len(self.r_mat)
        upperBounds = [0 for n in range(newNeuronNum+1)]
        lowerBounds = [0 for n in range(newNeuronNum+1)]
        upperBounds[-1] = np.inf
        lowerBounds[-1] = -np.inf
        weightMagMax = wMax
        weightMagMin = wMin
        if n < self.neuronNum // 2:
            for nIdx in range(newNeuronNum):
                if nIdx < newNeuronNum // 2:
                    upperBounds[nIdx] = weightMagMax
                    lowerBounds[nIdx] = 0
                    #bounds[nIdx] = (0, inf)
                else:
                    upperBounds[nIdx] = 0
                    lowerBounds[nIdx] = weightMagMin
                    #bounds[nIdx] = (None, 0)
        else:
            for nIdx in range(newNeuronNum):
                if nIdx < newNeuronNum // 2:
                    upperBounds[nIdx] = 0
                    lowerBounds[nIdx] = weightMagMin
                    #bounds[nIdx] = (None, 0)
                else:
                    upperBounds[nIdx] = weightMagMax
                    lowerBounds[nIdx] = 0
                    #bounds[nIdx] = (0, None)
        return (lowerBounds, upperBounds)
        #return bounds
    def RFitFunc(self, w_n, S, r):
        #x is w_i* with tonic attached to the end
        #y is s_e with extra 1 at the end
        #S must be nxe
        return abs(np.linalg.norm(np.dot(w_n, S) - r))
    def FitWeightMatrixExclude(self, fileLoc, eyeFileLoc, boundFunc=None):
        '''Fit fixed points in the network using target curves.

        Create an activation function matrix X (exn+1).
        Fit each row of the weight matrix with linear regression.
        Call the function to fit the predictor of eye position.
        Exclude eye positions where a neuron is at 0 for training each row.'''
        X = np.ones((len(self.eyePos), len(self.r_mat) + 1))
        for i in range(len(X)):
            for j in range(len(X[0]) - 1):
                X[i, j] = self.f(self.r_mat[j, i])
        """plt.plot(self.eyePos, X)
        plt.show()"""
        for n in range(self.neuronNum):
            if n%20==0:
                print(n)
            startIdx = int(self.cutoffIdx[n])
            # Do the fit
            # Two different because the two sides will want different sides of the matrix
            r = self.r_mat_neg[n]
            """solution = np.linalg.lstsq(X,r)[0]
            self.w_mat[n] = solution[:-1]
            self.T[n] = solution[-1]"""
            solution = None
            if boundFunc != None:
                bounds = boundFunc(n)
                solution = scipy.optimize.lsq_linear(X, r, bounds)
            else:
                solution = scipy.optimize.lsq_linear(X, r)
            self.w_mat[n] = solution.x[:-1]
            self.T[n] = solution.x[-1]
            #if n < self.neuronNum // 2:
                #r = self.r_mat_neg[n][startIdx:]
                #solution = np.linalg.lstsq(X[startIdx:, :], r)[0]
                #solution = np.linalg.lstsq(X,r)[0]
            #else:
                #r = self.r_mat_neg[n][:startIdx]
                #solution = np.linalg.lstsq(X[:startIdx, :], r)[0]
                #solution = np.linalg.lstsq(X,r)[0]
            """if n % 10 ==0:
                plt.plot(self.eyePos, np.dot(X[:,:-1], self.w_mat[n]) + self.T[n])
                plt.plot(self.eyePos, self.r_mat_neg[n])
                plt.ylim(-150, 150)
                plt.show()"""
        self.WriteWeightMatrix(solution.x, fileLoc)
        self.FitPredictorNonlinearSaturation(eyeFileLoc)

    #Depracated
    """def FitWeightMatrixMinimize(self):
        '''Fit fixed points in the network using target curves.

        Create an activation function matrix X (exn+1).
        Fit each row of the weight matrix with function minimization.
        Call the function to fit the predictor of eye position.'''
        X = np.ones((self.neuronNum+1, len(self.eyePos)))
        X[:-1,:] = self.f(self.r_mat)
        for n in range(len(self.r_mat)):
            #Set the bounds to be excitatory same side
            #and inhibitory to the opposite side.
            bounds = self.BoundQuadrants(n, .2, -.05)
            print(np.shape(bounds))
            #bounds = self.BoundDale(n)
            #Run the fit with the specified bounds
            guess = np.zeros((self.neuronNum + 1))
            func = lambda w_n: self.RFitFunc(w_n, X, self.r_mat[n])
            solution = sp.optimize.minimize(func, guess,bounds=bounds)
            self.w_mat[n] = solution.x[:-1]
            self.T[n] = solution.x[-1]
        self.FitPredictorNonlinearSaturation()"""
    def FitPredictorNonlinearSaturation(self, eyeWeightFileLoc):
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
        #print(self.predictW)
        self.predictT = weightSolution[-1]
        self.WriteWeightMatrix(weightSolution, eyeWeightFileLoc)
    def FitPredictorFacilitationLesion(self, sideRight):
        S_mat_T = np.ones((len(self.eyePos), self.neuronNum + 1))
        for i in range(len(S_mat_T)):
            for j in range(len(S_mat_T[0]) - 1):
                S_mat_T[i, j] = self.f(self.r_mat[j, i])
        if(sideRight):
            S_mat_T[:,0:self.neuronNum//2] = np.zeros((len(self.eyePos), self.neuronNum//2))
        else:
            S_mat_T[:,self.neuronNum//2:self.neuronNum] = np.zeros((len(self.eyePos), self.neuronNum // 2))
        # CHANGE: potentially include a tonic input in the prediction
        weightSolution = np.linalg.lstsq(S_mat_T, self.eyePos, rcond=None)[0]
        self.predictW = weightSolution[:-1]
        self.predictT = weightSolution[-1]

    def PredictEyePosNonlinearSaturation(self, s_E):
        '''Predict eye positions.

        Parameters
        s_E: a vector of activation at a given time point

        Returns predicted eye positions (constant).'''
        #print(s_E)
        #print(self.predictW)
        return np.dot(s_E, self.predictW) + self.predictT
    def GetR(self, s_e):
        '''Predict firing rates.

        Parameters
        s_E: a vector of activation at a given time point

        Returns
        r_E: a vector of firing rates at a given time point'''
        r_e = np.dot(self.w_mat, s_e) + self.T
        return r_e

    def GraphWeightMatrix(self):
        '''Create a heat map of the weight matrix (nxn).

        Rows are postsynaptic neurons. Columns are pre-synaptic neurons.'''
        plt.imshow(self.w_mat)
        plt.title("Weight Matrix")
        plt.colorbar()
        plt.show()
    def RunSim(self, startIdx=-1, dead=[]):
        '''Run simulation generating activation values.

        Set the starting value to the activation function of the target firing rates.
        Update using the update rule: t * ds/dt = -s + a*f(r).

        a*f(r) are wrapped in self.f()'''
        if not startIdx == -1:
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        #Set default values
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0])
        growthMat = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            for d in dead:
                r_vect[d] = 0
            decay = -self.s_mat[tIdx - 1]
            growth = self.f(r_vect)
            growthMat[tIdx] = growth
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            #Predict eye position based on synaptic activation
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            #Increment the time index
            tIdx += 1
        plt.plot(self.t_vect, growthMat)
        return eyePositions, growthMat
    def RunSimTau(self, alpha, startIdx=-1, plot=True, dead=[]):
        '''Run simulation generating activation values.

        Set the starting value to the activation function of the target firing rates.
        Update using the update rule: t * ds/dt = -s + a*(1-s)*r.'''
        if not startIdx == -1:
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        #Set default values
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0])
        growthMat = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            for r in range(len(r_vect)):
                if r_vect[r] < 0:
                    r_vect[r] = 0
            for d in dead:
                r_vect[d] = 0
            decay = -self.s_mat[tIdx - 1]
            growth = alpha * r_vect
            #Correct for the change in time constant due to division in the equation
            for g in range(len(growth)):
                growth[g] = (1-self.s_mat[tIdx-1][g]) * growth[g]
            growth = np.array(growth)
            growthMat[tIdx] = growth
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            #Predict eye position
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            #Increment the time index
            tIdx += 1
        return eyePositions
    def RunSimF(self, P0=.1, f=.1, t_f=50, startIdx=-1, dead=[], timeDead=100):
        '''Run simulation generating activation values. (Facilitation)

        Set the starting value to the activation function of the target firing rates.
        Update using the update rule: t * ds/dt = -s + P_rel * r.

        P_rel*r is wrapped in self.f()'''
        if not startIdx == -1:
            self.s_mat[0,:] = self.f(self.r_mat[:, startIdx]) #CHANGE
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0,:] = self.f(self.r_mat[:, startIdx]) #CHANGE
        #print("Printing s_mat")
        #print(self.s_mat[0,:])
        #Set default values
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0,:])
        P_rel = np.zeros((len(self.t_vect), self.neuronNum)) #NEW
        Rs = np.zeros((len(self.t_vect), self.neuronNum))  # NEW
        P0_vect = np.ones((self.neuronNum)) * P0
        P_rel[0] = np.array(ActivationFunction.SynapticFacilitationNoR(self.r_mat[:,startIdx], P0, f, t_f)) #NEW
        growthMat = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            if self.t_vect[tIdx] < timeDead:
                for d in dead:
                    r_vect[d] = 0
            Rs[tIdx]=r_vect
            #BIG CHANGE
            #print(r_vect/200)
            #r_Ca = np.power((r_vect/400),2)
            #print(0)
            #print(r_Ca)
            changeP = -P_rel[tIdx-1] + P0_vect + t_f*f*np.multiply(r_vect, (1-P_rel[tIdx-1]))
            #changeP = -P_rel[tIdx-1] + P0_vect + t_f*f*np.multiply(r_Ca, (1-P_rel[tIdx-1]))
            #print(1)
            #print(changeP)
            P_rel[tIdx] = P_rel[tIdx-1] + self.dt/t_f * changeP
            #print(2)
            #print(P_rel[tIdx])
            decay = -self.s_mat[tIdx - 1]
            #print(3)
            #print(decay)
            growth = np.multiply(P_rel[tIdx-1], r_vect)
            #print(4)
            #print(growth)
            growthMat[tIdx] = growth
            #print(5)
            #print(growthMat[tIdx])
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            #print(self.s_mat[tIdx])
            #Predict eye position based on synaptic activation
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            #print(eyePositions[tIdx])
            #Increment the time index
            tIdx += 1
        """plt.plot(self.t_vect, eyePositions, label="E")
        plt.plot(self.t_vect, P_rel[:,self.neuronNum//2-1]*25, label="P")
        plt.plot(self.t_vect, Rs[:,self.neuronNum//2-1], label="R")
        plt.plot(self.t_vect, self.s_mat[:,self.neuronNum//2-1], label="S")
        plt.legend()
        plt.show()"""
        return eyePositions, Rs
    def RunSimFCa(self, fixT_f, startIdx=-1, dead=[], timeDead=100):
        '''Run simulation generating activation values. (Facilitation)

        Set the starting value to the activation function of the target firing rates.
        Update using the update rule: t * ds/dt = -s + P_rel * r.

        P_rel*r is wrapped in self.f()'''
        if not startIdx == -1:
            self.s_mat[0,:] = self.f(self.r_mat[:, startIdx]) #CHANGE
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0,:] = self.f(self.r_mat[:, startIdx]) #CHANGE
        #print("Printing s_mat")
        #print(self.s_mat[0,:])
        #Set default values
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0,:])
        P_rel = np.zeros((len(self.t_vect), self.neuronNum)) #NEW
        Rs = np.zeros((len(self.t_vect), self.neuronNum))  # NEW
        P0_vect = np.ones((self.neuronNum)) * self.P0_f
        P_rel[0] = np.array(ActivationFunction.SynapticFacilitationNoR(self.r_mat[:,startIdx], self.P0_f, self.f_f, self.t_f)) #NEW
        growthMat = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            if self.t_vect[tIdx] < timeDead:
                for d in dead:
                    r_vect[d] = 0
            Rs[tIdx]=r_vect
            r_Ca = np.power(r_vect/self.r_Star,self.n_Ca)
            #changeP = -P_rel[tIdx-1] + P0_vect + t_f*f*np.multiply(r_vect, (1-P_rel[tIdx-1]))
            changeP = -P_rel[tIdx-1] + (P0_vect + self.t_f*self.f_f*r_Ca) / (1 + self.t_f * self.f_f * r_Ca)
            tau_eff = self.t_f/(1+self.t_f*self.f_f*r_Ca)
            #NEW
            if fixT_f > 0:
                tau_eff = fixT_f
            P_rel[tIdx] = P_rel[tIdx-1] + self.dt * np.multiply(1/tau_eff, changeP)
            decay = -self.s_mat[tIdx - 1]
            growth = np.multiply(P_rel[tIdx-1], r_vect)
            growthMat[tIdx] = growth
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            #Predict eye position based on synaptic activation
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            #Increment the time index
            tIdx += 1
        """plt.plot(self.t_vect, eyePositions, label="E")
        plt.plot(self.t_vect, P_rel[:,self.neuronNum//2-1]*25, label="P")
        plt.plot(self.t_vect, Rs[:,self.neuronNum//2-1], label="R")
        plt.plot(self.t_vect, self.s_mat[:,self.neuronNum//2-1], label="S")
        plt.legend()
        plt.show()"""
        return eyePositions, Rs
    def RunSimFCaRELU(self, startIdx=-1, dead=[], timeDead=100):
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
        #print(self.PredictEyePosNonlinearSaturation(self.s_mat[0,:]))
        try:
            eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0,:])
        except:
            print("Eye Position Prediction Error")
        P_rel = np.zeros((len(self.t_vect), self.neuronNum)) #NEW
        Rs = np.zeros((len(self.t_vect), self.neuronNum))  # NEW
        P0_vect = np.ones((self.neuronNum)) * self.P0_f
        P_rel[0] = np.array(ActivationFunction.SynapticFacilitationNoR(self.r_mat[:,startIdx], self.PO_f, self.f_f, self.t_f)) #NEW
        growthMat = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            if self.t_vect[tIdx] < timeDead:
                for d in dead:
                    r_vect[d] = 0
            Rs[tIdx]=r_vect
            r_Ca = np.where(r_vect-self.r0_f < 0, 0, r_vect-self.r0_f)
            changeP = -P_rel[tIdx-1] + P0_vect + self.t_f*self.f_f*np.multiply(r_Ca, (1-P_rel[tIdx-1]))
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
        """plt.plot(self.t_vect, eyePositions, label="E")
        plt.plot(self.t_vect, P_rel[:,self.neuronNum//2-1]*25, label="P")
        plt.plot(self.t_vect, Rs[:,self.neuronNum//2-1], label="R")
        plt.plot(self.t_vect, self.s_mat[:,self.neuronNum//2-1], label="S")
        plt.legend()
        plt.show()"""
        return eyePositions, Rs
    def RunSimFEye(self, startIdx=-1, dead=[], timeDead=100):
        '''Run simulation generating activation values. (Facilitation)

        Set the starting value to the activation function of the target firing rates.
        Update using the update rule: t * ds/dt = -s + P_rel * r.

        P_rel*r is wrapped in self.f()'''
        if not startIdx == -1:
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        #Set default values
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0])
        t_e = 500 #500ms time constant
        P_rel = np.zeros((len(self.t_vect), self.neuronNum)) #NEW
        Rs = np.zeros((len(self.t_vect), self.neuronNum))  # NEW
        P0_vect = np.ones((self.neuronNum)) * self.P0_f
        P_rel[0] = np.array(ActivationFunction.SynapticFacilitationNoR(self.r_mat[:,startIdx], self.PO_f, self.f_f, self.t_f))
        growthMat = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            if self.t_vect[tIdx] < timeDead:
                for d in dead:
                    r_vect[d] = 0
            t_r = 500
            #Rs[tIdx]=self.dt/t_r*(-Rs[tIdx-1] + r_vect)
            Rs[tIdx]=r_vect
            changeP = -P_rel[tIdx-1] + P0_vect + self.t_f*self.f_f*np.multiply(Rs[tIdx-1], (1-P_rel[tIdx-1]))
            P_rel[tIdx] = P_rel[tIdx-1] + self.dt/self.t_f * changeP
            decay = -self.s_mat[tIdx - 1]
            growth = np.multiply(P_rel[tIdx-1], Rs[tIdx-1])
            growthMat[tIdx] = growth
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            #Predict eye position based on synaptic activation
            eyePositions[tIdx] = eyePositions[tIdx-1] + self.dt/t_e *(-eyePositions[tIdx-1] + self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx]))
            #Increment the time index
            tIdx += 1
        return eyePositions, Rs
    def RunSimFEyeR(self, P0=.1, f=.4, t_f=50, t_r=500, startIdx=-1, dead=[], timeDead=100):
        '''Run simulation generating activation values. (Facilitation)

        Set the starting value to the activation function of the target firing rates.
        Update using the update rule: t * ds/dt = -s + P_rel * r.

        P_rel*r is wrapped in self.f()'''
        if not startIdx == -1:
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        #Set default values
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0])
        t_e = 500 #500ms time constant
        P_rel = np.zeros((len(self.t_vect), self.neuronNum)) #NEW
        Rs = np.zeros((len(self.t_vect), self.neuronNum))  # NEW
        P0_vect = np.ones((self.neuronNum)) * P0
        P_rel[0] = np.array(ActivationFunction.SynapticFacilitationNoR(self.r_mat[:,startIdx], P0, f, t_f)) #NEW
        growthMat = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            if self.t_vect[tIdx] < timeDead:
                for d in dead:
                    r_vect[d] = 0
            Rs[tIdx]=Rs[tIdx-1]+self.dt/t_r*(-Rs[tIdx-1] + r_vect)
            changeP = -P_rel[tIdx-1] + P0_vect + t_f*f*np.multiply(Rs[tIdx-1], (1-P_rel[tIdx-1]))
            P_rel[tIdx] = P_rel[tIdx-1] + self.dt/t_f * changeP
            decay = -self.s_mat[tIdx - 1]
            growth = np.multiply(P_rel[tIdx-1], Rs[tIdx-1])
            growthMat[tIdx] = growth
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            #Predict eye position based on synaptic activation
            eyePositions[tIdx] = eyePositions[tIdx-1] + self.dt/t_e *(-eyePositions[tIdx-1] + self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx]))
            #Increment the time index
            tIdx += 1
        return eyePositions, Rs
    def RunSimD(self, P0=.1, f=.4, t_f=50, startIdx=-1, dead=[]):
        '''Run simulation generating activation values. (Depression)

        Set the starting value to the activation function of the target firing rates.
        Update using the update rule: t * ds/dt = -s + P_rel * r.

        P_rel*r is wrapped in self.f()'''
        if not startIdx == -1:
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        #Set default values
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0])
        P_rel = np.zeros((len(self.t_vect), self.neuronNum)) #NEW
        Rs = np.zeros((len(self.t_vect), self.neuronNum))  # NEW
        P0_vect = np.ones((self.neuronNum)) * P0
        P_rel[0] = np.array(ActivationFunction.SynapticDepressionNoR(self.r_mat[:,startIdx], P0, f, t_f)) #NEW
        growthMat = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            Rs[tIdx]=r_vect
            for d in dead:
                self.s_mat[tIdx][d] = 0
            changeP = -P_rel[tIdx-1] + P0_vect - t_f*(1-f)*np.multiply(r_vect, P_rel[tIdx-1])
            P_rel[tIdx] = P_rel[tIdx-1] + self.dt/t_f * changeP
            decay = -self.s_mat[tIdx - 1]
            growth = np.multiply(P_rel[tIdx-1], r_vect)
            growthMat[tIdx] = growth
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            #Predict eye position based on synaptic activation
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            #Increment the time index
            tIdx += 1
        return eyePositions, growthMat
    def PlotFixedPointsOverEyePosRate(self,neuronArray):
        '''Plots synaptic activation decay and growth.

        Uses the prediction of firing rate vs actual firing rate
        to visualize fixed points of the network.'''
        for n in neuronArray:
            r = np.zeros(len(self.eyePos))
            for e in range(len(self.eyePos)):
                r[e] = np.dot(self.w_mat[n], self.f(self.r_mat[:,e])) + self.T[n]
                r[e] = max(0, r[e])
            #plt.plot(self.eyePos, self.r_mat[n], label = "decay")
            plt.plot(self.eyePos, r, label = "growth")
            plt.xlabel("Eye Position")
            plt.ylabel("Fixed Points")
    def PlotContribution(self):
        #Draw a line of the eye position
        #Right contribution is the weight matrix of the first half times the s of the first half
        #Left contribution is the opposite
        #Plot eye predicted, eye actual, left, and right
        zeros = np.zeros(len(self.eyePos)//2)
        eyeHalf = self.eyePos[len(self.eyePos)//2:]
        goalR = []
        for z in zeros:
            goalR.append(z)
        for e in eyeHalf:
            goalR.append(e)
        goalR = np.array(goalR)
        eR = np.zeros((len(self.eyePos)))
        eL = np.zeros((len(self.eyePos)))
        for e in range(len(self.eyePos)):
            sR = self.f(self.r_mat[:self.neuronNum//2,e])
            sL = self.f(self.r_mat[self.neuronNum//2:, e])
            eR[e] = np.dot(self.predictW[:self.neuronNum//2], sR)
            eL[e] = np.dot(self.predictW[self.neuronNum//2:], sL)
        plt.plot(self.eyePos, self.eyePos, label="Reference")
        plt.plot(self.eyePos, [self.PredictEyePosNonlinearSaturation(self.f(self.r_mat[:,e])) for e in range(len(self.eyePos))], label="Prediction")
        plt.plot(self.eyePos, eR, label="Right")
        plt.plot(self.eyePos, eL, label="Left")
        plt.plot(self.eyePos, [self.predictT for e in range(len(self.eyePos))], label="Tonic")
        plt.legend()
        plt.show()

        return np.average(np.square(np.subtract(goalR, eR)))
    #Network alterations
    def MistuneMatrix(self, fractionOffset = .01):
        self.w_mat = (1-fractionOffset) * self.w_mat
    """def PlotTauOverEyePos(self):
        interval = 500
        dead=[x for x in range(self.neuronNum//2, self.neuronNum)]
        ePlot = np.zeros((len(self.eyePos)//interval))
        figTau = plt.figure()
        figTau, axsTau = plt.subplots(2)
        tPlot = np.zeros((len(ePlot)))
        for e in range(len(self.eyePos)):
            if e%interval == 0:
                print(e)
                ePlot[int(e / interval)] = sim.eyePos[e]
                eyeVect, rVect = sim.RunSimFEye(P0, f, t_pGlobal, e, dead)
                axsTau[1].plot(self.t_vect, eyeVect)
                eyeStart = eyeVect[1]
                eyeStop = eyeVect[-1]
                #change[int(e / interval)] = eyeStart-eyeStop
                cutOffLower = .368 * (eyeStart - eyeStop)
                cutOffUpper = .632 * (eyeStop - eyeStart)
                #lowerBound[int(e / interval)] = cutOffLower
                newTau = 0
                #***BREAK POINT
                for t in range(len(self.t_vect)):
                    if eyeVect[t] < eyeStop + cutOffLower:
                        tPlot[int(e/interval)] = self.t_vect[t]
                        axsTau[1].scatter(self.t_vect[t], eyeVect[t])
                        break
                for t in range(len(self.t_vect)):
                    if eyeVect[t] > eyeStart + cutOffUpper:
                        tPlot[int(e / interval)] = self.t_vect[t]
                        axsTau[1].scatter(self.t_vect[t], eyeVect[t])
                        break
                #BREAK POINT END
        axsTau[0].plot(ePlot, tPlot)
        axsTau[0].set_xlabel("Eye Position Before Silencing")
        axsTau[0].set_ylabel("Time Constant (ms)")
        axsTau[1].set_xlabel("Time (ms)")
        axsTau[1].set_ylabel("Eye Position (degrees)")
        plt.tight_layout()
        plt.show()"""
    def GetTauVect(self, vect):
        errorIdx = 1
        vStart = vect[errorIdx]
        vStop = vect[-1]
        # change[int(e / interval)] = eyeStart-eyeStop
        cutOffLower = .368 * (vStart - vStop)
        cutOffUpper = .632 * (vStop - vStart)
        # lowerBound[int(e / interval)] = cutOffLower
        newTau = 0
        tIdx = 0
        if vStart > vStop:
            for t in range(errorIdx, len(self.t_vect)):
                if vect[t] < vStop + cutOffLower:
                    newTau = self.t_vect[t]
                    tIdx=t
                    break
        else:
            for t in range(errorIdx,len(self.t_vect)):
                if vect[t] > vStart + cutOffUpper:
                    newTau = self.t_vect[t]
                    tIdx=t
                    break
        return newTau, tIdx
    """def PlotTauOverEyePosRate(self):
        interval = 500
        dead=[x for x in range(self.neuronNum//2, 3*self.neuronNum//4)]
        bins = 15
        rBinnedAverage = [(0,0) for y in range(bins)]
        for e in [0]:#range(len(self.eyePos)):
            if e%interval == 0:
                eyeVect, rVect = sim.RunSimF(P0Global,fGlobal,t_pGlobal,e,dead)
                #plt.plot(sim.t_vect, rVect)
                for n in range(len(rVect[0])):
                    rStart = rVect[1,n]
                    rStop = rVect[-1,n]
                    #print((rStart,rStop))
                    cutOffLower = .368 * (rStart - rStop)
                    cutOffUpper = .632 * (rStop - rStart)
                    currTuple = rBinnedAverage[int(rStart // bins)]
                    for t in range(len(self.t_vect)):
                        if rStart > rStop: #Decay
                            if rVect[t,n] < rStop + cutOffLower:
                                #(size * average + value) / (size + 1)
                                #print(int(rStart//bins))
                                updatedAverage = (currTuple[1] * currTuple[0] + self.t_vect[t])/(currTuple[1]+1)
                                rBinnedAverage[int(rStart//bins)] = (updatedAverage, currTuple[1]+1)
                                #axsTau[1].scatter(self.t_vect[t], rVect[t])
                                break
                        else:
                            if rVect[t,n] > rStart + cutOffUpper:
                                #(size * average + value) / (size + 1)
                                #print(int(rStart//bins))
                                updatedAverage = (currTuple[1] * currTuple[0] + self.t_vect[t])/(currTuple[1]+1)
                                rBinnedAverage[int(rStart//bins)] = (updatedAverage, currTuple[1]+1)
                                break
        X = [str(int((i+1)*sim.maxFreq/bins)) for i in range(bins)]
        print(len(X))
        Y = [a[1] for a in rBinnedAverage]
        print(len(Y))
        plt.bar(X,Y)
        plt.title("Average Time Constant Over Firing Rate (Right Bin Labeled)")
        plt.show()
        return np.average(Y)"""
#External Functions
def GetDeadNeurons(fraction, firstHalf, neuronNum):
    if firstHalf:
        return [j for j in range(0, int(neuronNum // 2 * fraction))]
    else:
        return [neuronNum // 2 + j for j in range(0, int(neuronNum // 2 * fraction))]

#Define Simulation Parameters
#overlap = 5 #Degrees in which both sides of the brain are active
#neurons = 100 #Number of neurons simulated
#dt = .01

#My Nonlinearity: a=.4 p=1.4
"""alpha = 1
myNonlinearity = lambda r_vect: alpha * ActivationFunction.Geometric(r_vect, 10, 1.4)"""
#Paper Nonlinearity: a=80 p=1 (Works)
"""alpha = .4
myNonlinearity = lambda r_vect: alpha * ActivationFunction.Geometric(r_vect, 40, 1)"""
#Synaptic Saturation:
"""alpha = .05
myNonlinearity = lambda r_vect: ActivationFunction.SynapticSaturation(r_vect, alpha)"""
#Synaptic Facilitation:
"""P0Global = .1
fGlobal=.05
t_pGlobal=5000
myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, P0Global, fGlobal, t_pGlobal)"""

#f=.01 t=1000 P0=.1 worked for no restrictions
#P .1 f .001 t 1000
#P0Global=.1
#fGlobal = 1/100
#t_pGlobal = 20
#yNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitationCa(r_vect, P0Global, fGlobal, t_pGlobal, n_Ca, rStar=r_star)
#myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, P0Global, fGlobal, t_pGlobal)
"""P0Global=.1
fGlobal = .001
t_pGlobal = 1000
myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, P0Global, fGlobal, t_pGlobal)"""

#Synaptic Depression:
"""P0Global = .4
fGlobal=.996
t_pGlobal=100
myNonlinearity = lambda r_vect: ActivationFunction.SynapticDepression(r_vect, P0Global, fGlobal, t_pGlobal)"""

#Instantiate the simulation with correct parameters
#sim = Simulation(neurons, dt, 2000, 700, 150, -25, 25, 5000, myNonlinearity)
#dataLoc = "/Users/alex/Documents/Github/GoldmanLab/Code/EmreThresholdSlope_NatNeuroCells_All (1).xls"
#dataLoc = "/Users/alex/Documents/Github/GoldmanLab/Code/FakeTuningCurves.xlsx"
#sim = Simulation(neurons, dt, 2000, 20, 150, -25, 25, 5000, myNonlinearity, dataLoc)

#Create and plot the curves
#sim.PlotTargetCurves()

#Fit the weight matrix
#sim.FitWeightMatrixExclude(sim.BoundQuadrants)
#Need to graph over the whole of negative ranges too otherwise the ends that aren't trained could go positive and affect
#the results.

#Graph Weight Matrix
#sim.GraphWeightMatrix()

#Visualize fixed points
#sim.PlotFixedPointsOverEyePosRate(range(70))
#plt.show()
#sim.PlotContribution()
#Reverse Engineer Target Curves
#Plot a heat map of the accuracy of the prediction of firing rates
"""M = np.zeros((len(sim.eyePos), sim.neuronNum))
for e in range(len(sim.eyePos)):
    s = sim.f(sim.r_mat[:,e])
    r = np.dot(sim.w_mat, s) + sim.T
    #M[e] = r - sim.r_mat[:,e]
    plt.scatter(np.ones(len(r))*sim.eyePos[e], r)
plt.show()"""

"""plt.imshow(M)
plt.colorbar()
plt.show()"""

#*****Simulation Graphs Below*****
#Run simulations for every 500 eye position indices
#Plot a simulation with and without external input
"""fig = plt.figure()
fig, axs = plt.subplots(2)
#fig.suptitle("f(r) = r / (40 + r) ; a=.4")
#fig.suptitle("f(r) = 1 * r^1.4 / (10 + r^1.4)")
#fig.suptitle("f(r) = (P0 + f*r*t_P)/ (1 + r*f*t_P) ; P0=.1, f=.4, t_P=50ms")
fig.suptitle("f(r) = P0/(1 + (1-f)r*t_P)")
for e in range(len(sim.eyePos)):
    if e%500 == 0:
        print(e)
        #Choose between a regular simulation or a tau simulation(ONLY FOR a(1-s)r)
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 0, 0, 50, 200, 5000))
        e1, g = sim.RunSim(startIdx=e)
        axs[0].plot(sim.t_vect, e1)
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Eye Position")
        axs[0].set_title("No External Input")
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 10, 0, 50, 200, 5000))
        e2, g2 = sim.RunSim(startIdx=e)
        axs[1].plot(sim.t_vect, e2)
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Eye Position")
        axs[1].set_title("With External Input")
plt.tight_layout()
plt.show()"""

#Plot synaptic saturation simulation with and without external input
"""fig = plt.figure()
fig, axs = plt.subplots(2)
fig.suptitle("f(r) = a * (1-s) * r ; a=.05")
for e in range(len(sim.eyePos)):
    if e%500 == 0:
        print(e)
        #Choose between a regular simulation or a tau simulation(ONLY FOR a(1-s)r)
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 0, 0, 50, 200, 5000))
        e1, g = sim.RunSimTau(alpha=alpha, startIdx=e)
        axs[0].plot(sim.t_vect, e1)
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Eye Position")
        axs[0].set_title("No External Input")
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 10, 0, 50, 200, 5000))
        e2, g2 = sim.RunSim(startIdx=e)
        axs[1].plot(sim.t_vect, e2)
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Eye Position")
        axs[1].set_title("With External Input")
plt.tight_layout()
plt.show()"""

#Plot double dynamic equations with and without external input (Facilitation)
"""fig = plt.figure()
fig, axs = plt.subplots(2)
fig.suptitle("Dynamics for S and P_rel (Facilitation)")
for e in range(len(sim.eyePos)):
    if e%1000 == 0:
        print(e)
        #Choose between a regular simulation or a tau simulation(ONLY FOR a(1-s)r)
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 0, 0, 50, 200, sim.t_end))
        e1,p1 = sim.RunSimFCa(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e)
        axs[0].plot(sim.t_vect, e1)
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Eye Position")
        axs[0].set_title("No External Input")
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 20, 0, 10, 500, sim.t_end))
        e3,p3 = sim.RunSimFCa(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e)
        axs[1].plot(sim.t_vect, e3)
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Eye Position")
        axs[1].set_title("With External Input")
plt.tight_layout()
plt.show()"""
#Plot double dynamic equations with and without external input (Depression)
"""fig = plt.figure()
fig, axs = plt.subplots(2)
fig.suptitle("Dynamics for S and P_rel (Depression)")
for e in range(len(sim.eyePos)):
    if e%500 == 0:
        print(e)
        #Choose between a regular simulation or a tau simulation(ONLY FOR a(1-s)r)
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 0, 0, 50, 200, 5000))
        e1,p1 = sim.RunSimD(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e)
        axs[0].plot(sim.t_vect, e1)
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Eye Position")
        axs[0].set_title("No External Input")
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 10, 0, 50, 200, 5000))
        e3,p3 = sim.RunSimD(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e)
        axs[1].plot(sim.t_vect, e3)
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Eye Position")
        axs[1].set_title("With External Input")
plt.tight_layout()
plt.show()"""

#*****Lesion and Mistune Simulation Graphs Below*****
#sim.PlotTauOverEyePos()

#For a regular simulation
"""fig = plt.figure()
fig, axs = plt.subplots(3)
#fig.suptitle("f(r) = r / (40 + r) ; a=.4")
#fig.suptitle("f(r) = 1 * r^1.4 / (10 + r^1.4)")
fig.suptitle("f(r) = (P0 + f*r*t_P)/ (1 + r*f*t_P) ; P0=.1, f=.4, t_P=50ms (Steady State)")
#fig.suptitle("f(r) = P0/(1 + (1-f)r*t_P)
error = .01
numKilled = 4
for e in range(len(sim.eyePos)):
    if e%500 == 0:
        print(e)
        #Choose between a regular simulation or a tau simulation(ONLY FOR a(1-s)r)
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 0, 0, 50, 500, 5000))
        sim.MistuneMatrix(error)
        #Plot Mistuning
        #e1 = sim.RunSimTau(alpha, startIdx=e)
        e1,p1 = sim.RunSim(startIdx=e)
        axs[2].plot(sim.t_vect, e1)
        axs[2].set_xlabel("Time [ms]")
        axs[2].set_ylabel("Eye Position")
        axs[2].set_ylim((-30,30))
        axs[2].set_title("Mistune Error of " + str(error))
        sim.MistuneMatrix(-error) #Returns to the original matrix
        #Plot Lesion Left
        deadNeurons = [random.randint(0,neurons//2-1) for n in range(numKilled)]
        e2,p2 = sim.RunSim(startIdx=e, dead=deadNeurons)
        axs[0].plot(sim.t_vect, e2)
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Eye Position")
        axs[0].set_ylim((-30,30))
        axs[0].set_title("Lesion " + str(numKilled) + " Neurons Positive Slope")
        #Plot Lesion Right
        #e3 = sim.RunSimTau(alpha, startIdx=e)
        deadNeurons = [random.randint(neurons//2,neurons-1) for n in range(numKilled)]
        e3,p3 = sim.RunSim(startIdx=e, dead=deadNeurons)
        axs[1].plot(sim.t_vect, e3)
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Eye Position")
        axs[1].set_ylim((-30,30))
        axs[1].set_title("Lesion " + str(numKilled) + " Neurons Negative Slope")
plt.tight_layout()
plt.show()"""

#For a synaptic saturation simulation
"""fig = plt.figure()
fig, axs = plt.subplots(3)
fig.suptitle("f(r) = a * (1-s) * r ; a=.05")
error = .01
numKilled = 4
for e in range(len(sim.eyePos)):
    if e%500 == 0:
        print(e)
        #Choose between a regular simulation or a tau simulation(ONLY FOR a(1-s)r)
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 0, 0, 50, 500, 5000))
        sim.MistuneMatrix(error)
        #Plot Mistuning
        e1 = sim.RunSimTau(alpha, startIdx=e)
        axs[2].plot(sim.t_vect, e1)
        axs[2].set_xlabel("Time [ms]")
        axs[2].set_ylabel("Eye Position")
        axs[2].set_ylim((-30,30))
        axs[2].set_title("Mistune Error of " + str(error))
        sim.MistuneMatrix(-error) #Returns to the original matrix
        #Plot Lesion Left
        deadNeurons = [random.randint(0,neurons//2-1) for n in range(numKilled)]
        e2 = sim.RunSimTau(alpha, startIdx=e, dead=deadNeurons)
        axs[0].plot(sim.t_vect, e2)
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Eye Position")
        axs[0].set_ylim((-30,30))
        axs[0].set_title("Lesion " + str(numKilled) + " Neurons Positive Slope")
        #Plot Lesion Right
        #e3 = sim.RunSimTau(alpha, startIdx=e)
        deadNeurons = [random.randint(neurons//2,neurons-1) for n in range(numKilled)]
        e3 = sim.RunSimTau(alpha, startIdx=e, dead=deadNeurons)
        axs[1].plot(sim.t_vect, e3)
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Eye Position")
        axs[1].set_ylim((-30,30))
        axs[1].set_title("Lesion " + str(numKilled) + " Neurons Negative Slope")
plt.tight_layout()
plt.show()"""

#For a double dynamics simulation for relesase probablity
"""fig = plt.figure()
fig, axs = plt.subplots(3)
fig.suptitle("Dynamics for S and P_rel")
error = .01
numKilled = 50
for e in range(len(sim.eyePos)):
    if e%500 == 0:
        print(e)
        #Choose between a regular simulation or a tau simulation(ONLY FOR a(1-s)r)
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 0, 0, 50, 500, 5000))
        sim.MistuneMatrix(error)
        #Plot Mistuning
        #e1 = sim.RunSimTau(alpha, startIdx=e)
        e1,p1 = sim.RunSimF(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e, timeDead=3000)
        axs[2].plot(sim.t_vect, e1)
        axs[2].set_xlabel("Time [ms]")
        axs[2].set_ylabel("Eye Position")
        axs[2].set_ylim((-30,30))
        axs[2].set_title("Mistune Error of " + str(error))
        sim.MistuneMatrix(-error) #Returns to the original matrix
        #Plot Lesion Left
        #deadNeurons = [random.randint(0,neurons//2-1) for n in range(numKilled)]
        deadNeurons = GetDeadNeurons(.5, True)
        e2,p2 = sim.RunSimF(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e, dead=deadNeurons, timeDead=3000)
        axs[0].plot(sim.t_vect, e2)
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Eye Position")
        axs[0].set_ylim((-30,30))
        axs[0].set_title("Lesion " + str(numKilled) + " Neurons Positive Slope")
        #Plot Lesion Right
        #e3 = sim.RunSimTau(alpha, startIdx=e)
        #deadNeurons = [random.randint(neurons//2,neurons-1) for n in range(numKilled)]
        deadNeurons = GetDeadNeurons(.5, False)
        e3,p3 = sim.RunSimF(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e, dead=deadNeurons, timeDead=3000)
        axs[1].plot(sim.t_vect, e3)
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Eye Position")
        axs[1].set_ylim((-30,30))
        axs[1].set_title("Lesion " + str(numKilled) + " Neurons Negative Slope")
plt.tight_layout()"""

#*****Test Code Below*****
#Run simulations at each eye position for complete inactivation
"""fig = plt.figure()
fig, axs = plt.subplots(2)
fig.suptitle("Complete Lesion in a Facilitation Network")
myDead = [sim.neuronNum // 2 + j for j in range(0,sim.neuronNum // 2)]
sim.SetCurrentDoubleSplit(
    Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 0, 0, 50, 200, 5000))
for e in range(len(sim.eyePos)):#range(len(sim.eyePos)):
    if e%500 == 0:
        print(e)
        eyeVect, rVect = sim.RunSimF(P0Global,fGlobal,t_pGlobal,e,dead=myDead)
        axs[0].set_xlim(0,sim.t_end)
        axs[0].set_ylim(sim.eyeStart,sim.eyeStop)
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Eye Position (degrees)")
        axs[0].plot(sim.t_vect,eyeVect)
axs[0].set_title("Left Side Inactivation")

myDead = [j for j in range(0,sim.neuronNum // 2)]
for e in range(len(sim.eyePos)):
    if e%500 == 0:
        print(e)
        eyeVect, rVect = sim.RunSimF(P0Global,fGlobal,t_pGlobal,e,dead=myDead)
        axs[1].set_xlim(0,sim.t_end)
        axs[1].set_ylim(sim.eyeStart,sim.eyeStop)
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Eye Position (degrees)")
        axs[1].plot(sim.t_vect,eyeVect)
axs[1].set_title("Right Side Inactivation")
fig.tight_layout()
plt.show()"""

#Plot tau over eye position
#sim.PlotTauOverEyePos()

#Steadily inactivate neurons in a for loop
#Run a simulation for every set of damage
#Record how many eye positions are maihntained within the simulation script or by checking first - last
#Plot number of maintained eye positons as a function of damage
#Should notice a pattern where the number plateaus at about half the original values.
"""maintained = []
x = []
for i in range(10,neurons//2,10):
    print("Num killed: " + str(i))
    x.append(i)
    numMaintained = 0
    for e in range(len(sim.eyePos)):
        if e%200 == 0:
            print(e)
            myDead = [sim.neuronNum//2+j for j in range(i)]
            print(myDead)
            eyeVect, rVect = sim.RunSimF(P0Global,fGlobal,t_pGlobal,e,dead=myDead)
            plt.xlim(0,1000)
            plt.ylim(-25,25)
            plt.plot(sim.t_vect,eyeVect)
            if abs(eyeVect[-1]-eyeVect[0]) < 2:
                numMaintained += 1
    plt.show()
    print("Maintained: " + str(numMaintained))
    maintained.append(numMaintained)
plt.plot(x,maintained)
plt.show()"""

#Slowly lesion neurons from one side to observe when it starts to decay
#x = []
"""sim.FitPredictorFacilitationLesion(False)
for i in range(30,31):
    print("Num killed: " + str(i))
    #x.append(i)
    myDead = [sim.neuronNum//2+j for j in range(sim.neuronNum//2)]
    eyeVect, rVect = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 4000, dead=myDead)
    eyeVect2, rVect2 = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 100, dead=myDead)
    plt.plot(sim.t_vect, eyeVect)
    plt.plot(sim.t_vect, eyeVect2)
    plt.xlim(0,sim.t_end)
    plt.ylim(sim.eyeStart, sim.eyeStop)
plt.show()"""

#Plot the time constant of decay over different inactivation amounts at 2 eye positions
"""x = []
t100 = []
t4900 = []
for i in range(0,neurons//2,10):
    print("Num killed: " + str(i))
    x.append(i)
    myDead = [sim.neuronNum//2+j for j in range(i)]
    eyeVect, rVect = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 4900, dead=myDead)
    eyeVect2, rVect2 = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 100, dead=myDead)
    t100.append(sim.GetTauVect(eyeVect2))
    t4900.append(sim.GetTauVect(eyeVect))
    plt.plot(sim.t_vect, eyeVect)
    plt.plot(sim.t_vect, eyeVect2)
    plt.ylim(sim.eyeStart, sim.eyeStop)
plt.show()
plt.plot(x,t100,label="Eye Position " + str(int(sim.eyePos[100])))
plt.plot(x,t4900,label="Eye Position " + str(int(sim.eyePos[4900])))
plt.show()"""

#Plot the time constant of decay of the firing rates (GOAL)
"""tauTuples = [] #An array of tuples (starting firing rate, time constant)
tauEye = []
x = []
for e in range(len(sim.eyePos)):
    if e%1000 == 0:
        print(e)
        #Deactivate one side of the brain in its entirety
        #myDead = [sim.neuronNum//2 + j for j in range(sim.neuronNum//2)]
        myDead = [sim.neuronNum//2 + j for j in range(44)]
        #Run a simulation at an eye position and return a matrix of firing rates over time
        eyeVect, rVect = sim.RunSimF(P0Global, fGlobal,t_pGlobal, e, dead=myDead)
        #plt.plot(sim.t_vect, eyeVect, linewidth=4)
        #For each neuron, find the time constant
        for n in range(sim.neuronNum):
            n_r = rVect[:,n]
            nLesion_tau, tIdx = sim.GetTauVect(n_r)
            #plt.scatter(nLesion_tau, n_r[tIdx])
            tauTuples.append((n_r[1], nLesion_tau))
        thisTauEye = sim.GetTauVect(eyeVect)
        tauEye.append(thisTauEye[0])
        x.append(sim.eyePos[e])
        #plt.scatter(thisTauEye[0], eyeVect[thisTauEye[1]])
        #plt.show()
        #plt.plot(sim.t_vect, rVect)
        #plt.show()
plt.plot(x,tauEye)
plt.title("Eye Position Time Constant Over Eye Position (Left Inactivation)")
plt.xlabel("Eye Position (degrees)")
plt.ylabel("Time Constant (ms)")
plt.show()"""
#Now we have an array of time constants and starting firing rates
#Create an array of tuples (average, num) that stores a running average in each bin
"""bins = 15
rBinnedAverage = [(0,0) for y in range(bins)]
print(len(rBinnedAverage))
#For each tuple, find which bin it belongs to and add it to the final sum.
for tuple in tauTuples:
    avgTuple = rBinnedAverage[int(tuple[0]//bins)]
    updatedAverage = (avgTuple[0]*avgTuple[1] + tuple[1]) / (avgTuple[1] + 1)
    rBinnedAverage[int(tuple[0]// bins)] = (updatedAverage, avgTuple[1] + 1)
#Plot a bar chart of the average time constants at different firing rates
X = [sim.maxFreq/bins * i for i in range(len(rBinnedAverage))]
Y = [rBinnedAverage[i][0] for i in range(len(rBinnedAverage))]
print(X)
print(Y)
plt.bar(X,Y,width=10, align='edge',edgecolor="black")
plt.title("Averaged Time Constant Over Firing Rate")
plt.xlabel("Firing rate (Bins of 10Hz)")
plt.ylabel("Time Constant (ms)")
plt.show()"""

#Plot drift over eye position
"""x = []
drift = []
drift2 = []
simulationDuration = (sim.t_vect[-1] - sim.t_vect[0]) / 1000 #Seconds
driftSamplingInterval = 300
for e in range(len(sim.eyePos)):
    if e%driftSamplingInterval == 0:
        print(e)
        #Deactivate one side of the brain in its entirety
        myDead = [sim.neuronNum//2 + j for j in range(44)]
        #Run a simulation at an eye position and return a matrix of firing rates over time
        eyeVect, rVect = sim.RunSimF(P0Global, fGlobal,t_pGlobal, e, dead=myDead)
        drift.append((eyeVect[-1] - eyeVect[0]) / simulationDuration)
        x.append(sim.eyePos[e])
for e in range(len(sim.eyePos)):
    if e%driftSamplingInterval == 0:
        print(e)
        #Deactivate one side of the brain in its entirety
        myDead = [j+1 for j in range(44)]
        #Run a simulation at an eye position and return a matrix of firing rates over time
        eyeVect, rVect = sim.RunSimF(P0Global, fGlobal,t_pGlobal, e, dead=myDead)
        drift2.append((eyeVect[-1] - eyeVect[0]) / simulationDuration)
plt.plot(x,drift)
plt.plot(x,drift2)
plt.title("Drift Over Eye Position After Inactivation")
plt.xlabel("Eye Position (degrees)")
plt.ylabel("Drift (degrees/sec)")
plt.show()"""

#Demonstrate the sensitive neuron issue
#x = []
"""for i in range(0,neurons//2,4):
    print("Num killed: " + str(i))
    #x.append(i)
    myDead = [sim.neuronNum//2+j for j in range(sim.neuronNum//2) if j!=i]
    #eyeVect, rVect = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 4900, dead=myDead)
    eyeVect2, rVect2 = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 100, dead=myDead)
    #t100.append(sim.GetTauVect(eyeVect2))
    #t4900.append(sim.GetTauVect(eyeVect))
    #plt.plot(sim.t_vect, eyeVect, label=)
    plt.plot(sim.t_vect, eyeVect2, label=str(i) + " safe")
    plt.ylim(sim.eyeStart, sim.eyeStop)
plt.legend()
plt.show()"""

"""for i in range(0,neurons//2,4):
    print("Num killed: " + str(i))
    x.append(i)
    myDead = [j for j in range(sim.neuronNum//2) if j!=i]
    #eyeVect, rVect = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 4900, dead=myDead)
    eyeVect2, rVect2 = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 4900, dead=myDead)
    #t100.append(sim.GetTauVect(eyeVect2))
    #t4900.append(sim.GetTauVect(eyeVect))
    #plt.plot(sim.t_vect, eyeVect, label=)
    plt.plot(sim.t_vect, eyeVect2, label=str(i) + " safe")
    plt.ylim(sim.eyeStart, sim.eyeStop)
plt.legend()
plt.show()"""
#plt.plot(x,t100,label="Eye Position " + str(int(sim.eyePos[100])))
#plt.plot(x,t4900,label="Eye Position " + str(int(sim.eyePos[4900])))
#plt.show()

#*****Brute Force Tests*****
#42 Minute (Approx)
"""topFive = [(0,0,10000) for i in range(5)] #Array of the best five combinations stored as a tuple
for p in np.linspace(.1,1, 9):
    print(p)
    for f in np.linspace(.0001, .1, 10):
        print(f)
        myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, P0Global, fGlobal, t_pGlobal)
        sim = Simulation(neurons, dt, 2000, 50, 150, -25, 25, 5000, myNonlinearity)
        sim.FitWeightMatrixExclude(sim.BoundQuadrants)
        #Calculate the minimum distance from target line to predictions for L and R
        #Largest average eye position for left and right
        curr = sim.PlotContribution()
        temp = None
        for t in range(len(topFive)):
            if topFive[t][2] > curr:
                temp = topFive[t]
                topFive[t] = (p,f,curr)
                curr = temp[2]
print(topFive)
"""

