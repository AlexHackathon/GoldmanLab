import random

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize

import Helpers.ActivationFunction as ActivationFunction
import Helpers.CurrentGenerator


class Simulation:
    def __init__(self, neuronNum, dt, end, tau, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, nonLinearityFunction):
        self.neuronNum = neuronNum #Number of neurons in the simulation
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
        self.onPoints = np.append(np.linspace(self.eyeStart, overlap, self.neuronNum//2), np.linspace(-overlap, self.eyeStop, self.neuronNum//2))
        #Vector that marks where each neuron becomes positive
        self.cutoffIdx = np.zeros((self.neuronNum))
        self.eyePos = np.linspace(self.eyeStart, self.eyeStop, self.eyeRes) #Vector of eye positions

        self.current_mat = np.zeros((self.neuronNum, len(self.t_vect))) #Defaults to no current
        self.w_mat = np.zeros((self.neuronNum,self.neuronNum)) #nxn matrix of weights
        self.predictW = None #Weight vector used to predict eye position from firing rates
        self.predictT = None #Tonic input for adjustments to the predicted eye position
        self.s_mat = np.zeros((len(self.t_vect), self.neuronNum)) #txn 2d array for storing information from the simulations
        self.T = np.zeros((self.neuronNum,)) #Tonic input to all the neurons
        self.r_mat =  np.zeros((self.neuronNum, len(self.eyePos)))#Tuning curves created for training the network's fixed points
        self.r_mat_neg = np.zeros((self.neuronNum, len(self.eyePos)))  # Tuning curves negative created for training the network's fixed points
        self.CreateTargetCurves()
        self.CreateTargetCurvesNeg()
    def SetCurrent(self, currentMat):
        '''Sets the current to a matrix (nxt).'''
        self.current_mat = currentMat
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
    def PlotTargetCurves(self):
        '''Plot given target curves over eye position.'''
        for r in range(len(self.r_mat)):
            plt.plot(self.eyePos, self.r_mat[r])
        plt.xlabel("Eye Position")
        plt.ylabel("Firing Rate")
        plt.show()
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
    def BoundQuadrants(self, n):
        '''Return a vector of restrictions based on same side excitation
        and opposite side inhibition.'''
        upperBounds = [0 for n in range(self.neuronNum+1)]
        lowerBounds = [0 for n in range(self.neuronNum+1)]
        upperBounds[-1] = np.inf
        lowerBounds[-1] = -np.inf
        weightMagMax = .04
        if n < self.neuronNum // 2:
            for nIdx in range(self.neuronNum):
                if nIdx < self.neuronNum // 2:
                    upperBounds[nIdx] = weightMagMax
                    lowerBounds[nIdx] = 0
                    #bounds[nIdx] = (0, inf)
                else:
                    upperBounds[nIdx] = 0
                    lowerBounds[nIdx] = -weightMagMax
                    #bounds[nIdx] = (None, 0)
        else:
            for nIdx in range(self.neuronNum):
                if nIdx < self.neuronNum // 2:
                    upperBounds[nIdx] = 0
                    lowerBounds[nIdx] = -weightMagMax
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
    def FitWeightMatrix(self):
        '''Fit fixed points in the network using target curves.

        Create an activation function matrix X (exn+1).
        Fit each row of the weight matrix with linear regression.
        Call the function to fit the predictor of eye position.'''
        """X = np.ones((len(self.eyePos), self.neuronNum + 1))
        for i in range(len(X)):
            for j in range(len(X[0]) - 1):
                X[i, j] = self.f(self.r_mat[j, i])"""
        X = self.f(np.transpose(self.r_mat))
        X = np.append(X, np.ones((len(self.eyePos),1)), axis=1)
        for n in range(self.neuronNum):
            r = self.r_mat[n]
            solution = np.linalg.lstsq(X, r)[0]
            self.w_mat[n] = solution[:-1]
            self.T[n] = solution[-1]
        self.FitPredictorNonlinearSaturation()
    def FitWeightMatrixExclude(self, boundFunc=None):
        '''Fit fixed points in the network using target curves.

        Create an activation function matrix X (exn+1).
        Fit each row of the weight matrix with linear regression.
        Call the function to fit the predictor of eye position.
        Exclude eye positions where a neuron is at 0 for training each row.'''
        X = np.ones((len(self.eyePos), self.neuronNum + 1))
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
        self.FitPredictorNonlinearSaturation()
    def FitWeightMatrixMinimize(self):
        '''Fit fixed points in the network using target curves.

        Create an activation function matrix X (exn+1).
        Fit each row of the weight matrix with function minimization.
        Call the function to fit the predictor of eye position.'''
        X = np.ones((self.neuronNum+1, len(self.eyePos)))
        X[:-1,:] = self.f(self.r_mat)
        for n in range(self.neuronNum):
            #Set the bounds to be excitatory same side
            #and inhibitory to the opposite side.
            bounds = self.BoundQuadrants(n)
            #bounds = self.BoundDale(n)
            #Run the fit with the specified bounds
            guess = np.zeros((self.neuronNum + 1))
            func = lambda w_n: self.RFitFunc(w_n, X, self.r_mat[n])
            solution = sp.optimize.minimize(func, guess,bounds=bounds)
            self.w_mat[n] = solution.x[:-1]
            self.T[n] = solution.x[-1]
        self.FitPredictorNonlinearSaturation()
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
    def RunSimF(self, P0=.1, f=.4, t_f=50, startIdx=-1, dead=[]):
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
        P_rel = np.zeros((len(self.t_vect), self.neuronNum)) #NEW
        Rs = np.zeros((len(self.t_vect), self.neuronNum))  # NEW
        P0_vect = np.ones((self.neuronNum)) * P0
        P_rel[0] = np.array(ActivationFunction.SynapticFacilitationNoR(self.r_mat[:,startIdx], P0Global, fGlobal, t_pGlobal)) #NEW
        growthMat = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            for d in dead:
                r_vect[d] = 0
            Rs[tIdx]=r_vect
            changeP = -P_rel[tIdx-1] + P0_vect + t_f*f*np.multiply(r_vect, (1-P_rel[tIdx-1]))
            P_rel[tIdx] = P_rel[tIdx-1] + self.dt/t_f * changeP
            decay = -self.s_mat[tIdx - 1]
            growth = np.multiply(P_rel[tIdx-1], r_vect)
            growthMat[tIdx] = growth
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            for d in dead:
                self.s_mat[tIdx][d] = 0
            #Predict eye position based on synaptic activation
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
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
        P_rel[0] = np.array(ActivationFunction.SynapticDepressionNoR(self.r_mat[:,startIdx], P0Global, fGlobal, t_pGlobal)) #NEW
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
            plt.plot(self.eyePos, self.r_mat[n], label = "decay")
            plt.plot(self.eyePos, r, label = "growth")
            plt.xlabel("Eye Position")
            plt.ylabel("Fixed Points")

    #Network alterations
    def MistuneMatrix(self, fractionOffset = .01):
        self.w_mat = (1-fractionOffset) * self.w_mat

    def PlotTauOverEyePos(self):
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
                eyeVect, rVect = sim.RunSimF(P0Global,fGlobal,t_pGlobal,e,dead)
                axsTau[1].plot(self.t_vect, eyeVect)
                eyeStart = eyeVect[1]
                eyeStop = eyeVect[-1]
                #change[int(e / interval)] = eyeStart-eyeStop
                cutOffLower = .368 * (eyeStart - eyeStop)
                cutOffUpper = .632 * (eyeStop - eyeStart)
                #lowerBound[int(e / interval)] = cutOffLower
                newTau = 0
                """for t in range(len(self.t_vect)):
                    if eyeVect[t] < eyeStop + cutOffLower:
                        tPlot[int(e/interval)] = self.t_vect[t]
                        axsTau[1].scatter(self.t_vect[t], eyeVect[t])
                        break"""
                for t in range(len(self.t_vect)):
                    if eyeVect[t] > eyeStart + cutOffUpper:
                        tPlot[int(e / interval)] = self.t_vect[t]
                        axsTau[1].scatter(self.t_vect[t], eyeVect[t])
                        break
        axsTau[0].plot(ePlot, tPlot)
        plt.show()
    def GetTauEyePos(self, eyeVect):
        eyeStart = eyeVect[1]
        eyeStop = eyeVect[-1]
        # change[int(e / interval)] = eyeStart-eyeStop
        cutOffLower = .368 * (eyeStart - eyeStop)
        cutOffUpper = .632 * (eyeStop - eyeStart)
        # lowerBound[int(e / interval)] = cutOffLower
        newTau = 0
        if eyeStart > eyeStop:
            for t in range(len(self.t_vect)):
                if eyeVect[t] < eyeStop + cutOffLower:
                    newTau = self.t_vect[t]
                    break
        else:
            for t in range(len(self.t_vect)):
                if eyeVect[t] > eyeStart + cutOffUpper:
                    newTau = self.t_vect[t]
                    break
        return newTau
    def PlotTauOverEyePosRate(self):
        interval = 500
        dead=[x for x in range(self.neuronNum//2, 3*self.neuronNum//4)]
        bins = 15
        rBinnedAverage = [(0,0) for y in range(bins)]
        for e in range(len(self.eyePos)):
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
        return np.average(Y)

overlap = 5 #Degrees in which both sides of the brain are active
neurons = 90 #Number of neurons simulated
dt = .01
#(self, neuronNum, dt, end, tau, a, p, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, nonlinearityFunction):

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

P0Global = .01
#fGlobal=.3
fGlobal = .01
t_pGlobal = 1000
myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, P0Global, fGlobal, t_pGlobal)

#Synaptic Depression:
"""P0Global = .4
fGlobal=.996
t_pGlobal=100
myNonlinearity = lambda r_vect: ActivationFunction.SynapticDepression(r_vect, P0Global, fGlobal, t_pGlobal)"""

#Instantiate the simulation with correct parameters
sim = Simulation(neurons, dt, 2000, 50, 150, -25, 25, 5000, myNonlinearity)

#Create and plot the curves
#sim.PlotTargetCurves()

#Fit the weight matrix
sim.FitWeightMatrixExclude()
#Need to graph over the whole of negative ranges too otherwise the ends that aren't trained could go positive and affect
#the results.

#Graph Weight Matrix
"""sim.GraphWeightMatrix()"""

#Visualize fixed points
"""sim.PlotFixedPointsOverEyePosRate(range(neurons))
plt.show()
"""
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
    if e%500 == 0:
        print(e)
        #Choose between a regular simulation or a tau simulation(ONLY FOR a(1-s)r)
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 0, 0, 50, 200, 5000))
        e1,p1 = sim.RunSimF(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e)
        axs[0].plot(sim.t_vect, e1)
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Eye Position")
        axs[0].set_title("No External Input")
        sim.SetCurrentDoubleSplit(
            Helpers.CurrentGenerator.ConstCurrentParameterized(sim.t_vect, dt, 10, 0, 50, 200, 5000))
        e3,p3 = sim.RunSimF(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e)
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
        e1,p1 = sim.RunSimF(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e)
        axs[2].plot(sim.t_vect, e1)
        axs[2].set_xlabel("Time [ms]")
        axs[2].set_ylabel("Eye Position")
        axs[2].set_ylim((-30,30))
        axs[2].set_title("Mistune Error of " + str(error))
        sim.MistuneMatrix(-error) #Returns to the original matrix
        #Plot Lesion Left
        deadNeurons = [random.randint(0,neurons//2-1) for n in range(numKilled)]
        e2,p2 = sim.RunSimF(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e, dead=deadNeurons)
        axs[0].plot(sim.t_vect, e2)
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Eye Position")
        axs[0].set_ylim((-30,30))
        axs[0].set_title("Lesion " + str(numKilled) + " Neurons Positive Slope")
        #Plot Lesion Right
        #e3 = sim.RunSimTau(alpha, startIdx=e)
        deadNeurons = [random.randint(neurons//2,neurons-1) for n in range(numKilled)]
        e3,p3 = sim.RunSimF(P0=P0Global, f=fGlobal, t_f=t_pGlobal, startIdx=e, dead=deadNeurons)
        axs[1].plot(sim.t_vect, e3)
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Eye Position")
        axs[1].set_ylim((-30,30))
        axs[1].set_title("Lesion " + str(numKilled) + " Neurons Negative Slope")
plt.tight_layout()
plt.show()"""

#*****Test Code Below*****
#Run simulations at one eye positions to find best alpha
"""minChange = 1000
minAlpha = 1000
alphaRes = 20
for alpha in np.linspace(0.0001,1,alphaRes):
    print(alpha)
    myNonlinearity = lambda r_vect: alpha * ActivationFunction.Geometric(r_vect, 20, 1)
    sim = Simulation(neurons, .01, 2000, 20, 150, -25, 25, 2000, myNonlinearity)
    sim.CreateTargetCurves()
    sim.FitWeightMatrix()
    print("Fit")
    change = sim.RunSim((5,0,50, 200))
    print(change)
    plt.scatter(alpha, change)
    if change < minChange:
        minChange = change
        minAlpha = alpha
plt.show()
print(minChange)
print(minAlpha)"""

#Find the optimal tau and f using brute force
"""X=np.zeros((25,50))
row = 0
column = 0
for p in np.linspace(.000001,.3, 25):
    print(p)
    for f in np.linspace(0,1,50):
        P0Global = p
        fGlobal = f
        t_pGlobal = 2000
        myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, P0Global, fGlobal, t_pGlobal)
        sim.f = myNonlinearity
        sim.FitWeightMatrixExclude()
        X[row][column] = sim.PlotTauOverEyePosRate()
        column = column+1
    row = row+1"""

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
plt.show()
"""

#Slowly lesion neurons from one side to observe when it starts to decay
#x = []
"""for i in range(0,neurons//2,4):
    print("Num killed: " + str(i))
    #x.append(i)
    myDead = [sim.neuronNum//2+j for j in range(i)]
    eyeVect, rVect = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 4900, dead=myDead)
    eyeVect2, rVect2 = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 100, dead=myDead)
    plt.plot(sim.t_vect, eyeVect)
    plt.plot(sim.t_vect, eyeVect2)
    plt.xlim(0,1000)
    plt.ylim(-25, 25)
plt.show()"""

#Plot the time constant of decay over different eye positions
x = []
t100 = []
t4900 = []
for i in range(0,neurons//2,10):
    print("Num killed: " + str(i))
    x.append(i)
    myDead = [sim.neuronNum//2+j for j in range(i)]
    eyeVect, rVect = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 4900, dead=myDead)
    eyeVect2, rVect2 = sim.RunSimF(P0Global, fGlobal, t_pGlobal, 100, dead=myDead)
    t100.append(sim.GetTauEyePos(eyeVect2))
    t4900.append(sim.GetTauEyePos(eyeVect))
    plt.plot(sim.t_vect, eyeVect)
    plt.plot(sim.t_vect, eyeVect2)
    plt.ylim(sim.eyeStart, sim.eyeStop)
plt.show()
plt.plot(x,t100,label="Eye Position " + str(int(sim.eyePos[100])))
plt.plot(x,t4900,label="Eye Position " + str(int(sim.eyePos[4900])))
plt.show()
#sim.PlotTauOverEyePosRate()
