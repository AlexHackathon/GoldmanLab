import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import Helpers.ActivationFunction as ActivationFunction

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
        self.CreateTargetCurves()
    def SetCurrent(self, currentMat):
        '''Sets the current to a matrix (nxt).'''
        self.current_mat = currentMat
    def SetWeightMatrix(self, weightMatrix):
        '''Sets the weight matrix to the given matrix (nxn).'''
        self.w_mat = weightMatrix
    def CreateTargetCurvesNeg(self):
        '''Create target tuning curves.

        Calculate slope given self.eyeStart, self.eyeStop, self.maxFreq.
        Create the line for a neuron based on the x intercept given by self.onPoints.
        Mark indicies where a neuron begins to be non-zero or begins to be zero (based on side).'''
        slope = self.maxFreq / (self.eyeStop - self.eyeStart)
        for n in range(self.neuronNum):
            for eIdx in range(len(self.eyePos)):
                #If neurons have positive slope and have 0s at the start
                y = None
                if n < self.neuronNum//2:
                    y = slope * (self.eyePos[eIdx] - self.onPoints[n])
                #If neurons have negative slope and end with 0s
                else:
                    y = -slope * (self.eyePos[eIdx] - self.onPoints[n])
                self.r_mat[n][eIdx] = y
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
        bounds = [0 for n in range(self.neuronNum + 1)]
        bounds[-1] = (None, None)
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
        '''Return a vector of restrictions based on same side excitation
        and opposite side inhibition.'''
        bounds = [0 for n in range(self.neuronNum + 1)]
        bounds[-1] = (None, 0)
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
        X = np.ones((len(self.eyePos), self.neuronNum + 1))
        for i in range(len(X)):
            for j in range(len(X[0]) - 1):
                X[i, j] = self.f(self.r_mat[j, i])
        for n in range(self.neuronNum):
            r = self.r_mat[n]
            solution = np.linalg.lstsq(X, r)[0]
            self.w_mat[n] = solution[:-1]
            self.T[n] = solution[-1]
        self.FitPredictorNonlinearSaturation()
    def FitWeightMatrixExclude(self):
        '''Fit fixed points in the network using target curves.

        Create an activation function matrix X (exn+1).
        Fit each row of the weight matrix with linear regression.
        Call the function to fit the predictor of eye position.
        Exclude eye positions where a neuron is at 0 for training each row.'''
        X = np.ones((len(self.eyePos), self.neuronNum + 1))
        for i in range(len(X)):
            for j in range(len(X[0]) - 1):
                X[i, j] = self.f(self.r_mat[j, i])
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
            print(n)
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
    def RunSim(self, current, startIdx=-1, plot=True, dead=[]):
        '''Run simulation generating activation values.

        Set the starting value to the activation function of the target firing rates.
        Update using the update rule: t * ds/dt = -s + a*f(r).

        a*f(r) are wrapped in self.f()'''
        mag = None
        currStart = None
        dur = None
        soa = None
        try:
            mag, currStart, dur, soa = current
            currStart = int(currStart)
            dur = int(dur)
            soa = int(soa)
        except:
            raise Exception("current was not of the specified format: (mag, currStart, dur, soa)")
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
        #Rs = np.zeros((len(self.t_vect),self.neuronNum))
        #Rs[0] = self.r_mat[:,startIdx] #CHANGE
        deltas = np.zeros((len(self.t_vect), self.neuronNum))
        while tIdx < len(self.t_vect):
            # Create a current starting at (currStart)ms lasting (dur)ms reoccurring every (soa)ms
            current = np.zeros((self.neuronNum))
            if tIdx > currStart and mag != 0:
                if tIdx % (soa / self.dt) >= 0 and tIdx % (soa / self.dt) < (dur / self.dt):
                    for n in range(self.neuronNum):
                        if self.s_mat[tIdx-1,n]==0:
                            continue
                        if n < self.neuronNum // 2:
                            current[n] = mag
                        else:
                            current[n] = -mag
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + current)
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            for d in dead:
                r_vect[d] = 0
            decay = -self.s_mat[tIdx - 1]
            growth = self.f(r_vect)
            deltas[tIdx] = decay + growth
            #Rs[tIdx] = r_vect #CHANGE
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            #Predict eye position based on synaptic activation
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            #Increment the time index
            tIdx += 1
        #Plot a graph of eye position over time if required.
        if plot:
            #plt.imshow(deltas, aspect="auto")
            #plt.colorbar()
            #plt.show()
            plt.plot(self.t_vect, eyePositions, label="Eye Position")
            #plt.plot(self.t_vect, Rs[:,20], label="Increasing")
            #plt.plot(self.t_vect, Rs[:,90], label="Decreasing")
            #plt.legend()
            #plt.show()
            #plt.xlabel("Time (ms)")
            #plt.ylabel("Eye Position (degrees)")
            #plt.plot(self.t_vect, deltas)
        plt.xlabel("Time (ms)")
        plt.ylabel("Eye Position (degrees)")
    def RunSimTau(self, startIdx=-1, plot=True, dead=[]):
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
        while tIdx < len(self.t_vect):
            #Create a current lasting (dur)ms starting every (soa)ms
            current = np.zeros((self.neuronNum))
            mag = 0
            soa = 1000
            dur = 300
            if tIdx % (soa / self.dt) >= 0 and tIdx % (soa / self.dt) < (dur / self.dt):
                for n in range(self.neuronNum):
                    if n < self.neuronNum // 2:
                        current[n] = mag
                    else:
                        current[n] = -mag
            #Calculate firing rates and prevent negative values
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + current)
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
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            #Predict eye position
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            #Increment the time index
            tIdx += 1
        #Plot a graph of eye position over time if required.
        if plot:
            plt.plot(self.t_vect, eyePositions)
        plt.xlabel("Time (ms)")
        plt.ylabel("Eye Position (degrees)")
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


overlap = 0 #Degrees in which both sides of the brain are active
neurons = 400 #Number of neurons simulated
#(self, neuronNum, dt, end, tau, a, p, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, nonlinearityFunction):
#alpha = .01 #Quadratic
#alpha = 1 #Geometric
alpha = .05 #Synaptic, Geometric Paper

#Change the nonlinearity of the simulation
#myNonlinearity = lambda r_vect: ActivationFunction.SynapticSaturation(r_vect, alpha)
#myNonlinearity = lambda r_vect: alpha * ActivationFunction.Geometric(r_vect, 20, 1)
myNonlinearity = lambda r_vect: alpha * ActivationFunction.Geometric(r_vect, .4, 1.4)
#myNonlinearity = lambda  r_vect: alpha * (np.multiply(r_vect,r_vect))
#myNonlinearity = lambda r_vect: ActivationFunction.SynapticFacilitation(r_vect, .1, .4, 50)

#Instantiate the simulation with correct parameters
sim = Simulation(neurons, .01, 1000, 20, 150, -25, 25, 2000, myNonlinearity)

#Create and plot the curves
sim.CreateTargetCurves()
#sim.CreateTargetCurvesNeg()
sim.PlotTargetCurves()

#Fit the weight matrix
sim.FitWeightMatrix() #Works well with: SynapticFacilitation
#sim.FitWeightMatrixExclude()

#Graph Weight Matrix
"""plt.imshow(sim.w_mat)
plt.colorbar()
plt.show()"""

#Visualize fixed points
sim.PlotFixedPointsOverEyePosRate(range(neurons))
plt.show()

#Reverse Engineer Target Curves
#Plot a heat map of the accuracy of the prediction of firing rates
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

#Plot the predicted firing rates as a scatterplot
"""for e in range(len(sim.eyePos)):
    predict = sim.PredictEyePosNonlinearSaturation(sim.f(sim.r_mat[:,e]))
    plt.scatter(sim.eyePos[e], predict)
plt.show()"""

#Run simulations for every 100 eye position indices
for e in range(len(sim.eyePos)):
    if e%200 == 0 and e < 1000:
        print(sim.eyePos[e])
        #Choose between a regular simulation or a tau simulation(ONLY FOR a(1-s)r)
        #sim.RunSim((0,0,0,0), startIdx=e)
        sim.RunSim((3,500,50,1000), startIdx=e)
        #plt.show()
plt.show()