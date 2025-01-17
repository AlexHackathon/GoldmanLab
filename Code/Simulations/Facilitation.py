import numpy as np
import pickle
class FacilitationSim:
    def __init__(self, dt, end, t_s, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, nonLinearityFunction, nonlinearityNoR):
        #self.neuronNum = neuronNum #Number of neurons in the simulation
        self.dt = dt #Time step [ms]
        self.t_end = end #Simulation end [ms]
        self.t_vect = np.arange(0, self.t_end, self.dt) #A time vector ranging from 0 to self.t_end

        self.eyeStart = eyeStartParam #Start of eye positions (degrees)
        self.eyeStop = eyeStopParam #End of eye positions (degrees)
        self.eyeRes = eyeResParam #Number of points between the start and end
        self.maxFreq = maxFreq #Highest frequency reached by a neuron

        self.t_s = t_s #Time constant
        self.f_noR = nonlinearityNoR #The nonlinearity used in the network not multiplied by firing rate
        self.f = nonLinearityFunction #The sole nonlinearity used in this network

        self.eyePos = np.linspace(self.eyeStart, self.eyeStop, self.eyeRes) #Vector of eye positions
        self.r_mat = None
        self.r_mat_neg = None
        self.cutoffIdx = None
        self.neuronNum = len(self.r_mat)
        self.w_mat = np.zeros((self.neuronNum,self.neuronNum)) #nxn matrix of weights
        self.current_mat = np.zeros((self.neuronNum, len(self.t_vect)))  # Defaults to no current
        self.predictW = None  # Weight vector used to predict eye position from firing rates
        self.predictT = None  # Tonic input for adjustments to the predicted eye position
        self.s_mat = np.zeros((len(self.t_vect), self.neuronNum))  # txn 2d array for storing information from the simulations
        self.T = np.zeros((self.neuronNum,))  # Tonic input to all the neurons

        #Facilitation Threshold Variables
        self.f_f = .4
        self.t_f = 500
        self.P0_f = .1
    def SetWeightMatrix(self, weightMatrix):
        '''Sets the weight matrix to the given matrix (nxn).'''
        self.w_mat = weightMatrix

    def WriteWeightMatrix(self, matrix, fileName):
        pickle.dump(matrix, open(fileName, "wb"))
    def RunFacilitation(self, predictionFunc, debug=True, startIdx=-1, dead=[], timeDead=100000000):
        '''Run simulation generating activation values. (Facilitation)

        Set the starting value to the activation function of the target firing rates.
        Update using the update rule: t * ds/dt = -s + P_rel * r.

        P_rel*r is wrapped in self.f()'''
        #Print all default values
        if(debug):
            print("startIdx: " + str(startIdx))
            print("dead: " + str(dead))
            print("timeDead: " + str(timeDead))

        #Sets the eye position to 0 or a set index of eye position
        if not startIdx == -1:
            # Set eye position to 0
            self.s_mat[0,:] = self.f(self.r_mat[:, startIdx])
        else:
            #Set eye position to the last eye position
            startIdx = self.neuronNum // 2
            self.s_mat[0,:] = self.f(self.r_mat[:, startIdx])
        #Set up simulation storage variables
        tIdx = 1
        # Eye positions of the simulation
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0,:])
        #Firing rates of all neurons throughout the simulation
        rSimMat = np.zeros((len(self.t_vect), self.neuronNum))
        #Vector of P0 for ease of multiplication
        P0_vect = np.ones((self.neuronNum)) * self.P0_f
        #Vector of all Prel values throughout the simulation
        P_rel = np.zeros((len(self.t_vect), self.neuronNum))
        P_rel[0] = np.array(self.f_noR(self.r_mat[:,startIdx], self.P0_f, self.f_f, self.t_f))
        while tIdx < len(self.t_vect):
            #Calculate firing rates and prevent negative values
            #r = Ws + T
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + self.current_mat[:,tIdx-1])
            r_vect = np.array([0 if r < 0 else r for r in r_vect])
            #Remove the firing rate of the dead neurons
            if self.t_vect[tIdx] < timeDead:
                for d in dead:
                    r_vect[d] = 0
            rSimMat[tIdx]=r_vect

            #Calculate dPrel
            #t_f * dPrel/dt = -Prel + P0 + t_f * f_f * r * (1-Prel)
            changeP = -P_rel[tIdx-1] + P0_vect + self.t_f*self.f_f* np.multiply(r_vect, 1-P_rel[tIdx-1])
            P_rel[tIdx] = P_rel[tIdx-1] + self.dt/self.t_f * changeP

            #Calculate ds
            #t_s * ds/dt = -s + Prel * r
            decay = -self.s_mat[tIdx - 1]
            growth = np.multiply(P_rel[tIdx-1], r_vect)
            #Update with the synaptic activation with the update rule
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt / self.t_s * (decay + growth)
            #Predict eye position based on synaptic activation
            eyePositions[tIdx] = predictionFunc(self.s_mat[tIdx]) #CHANGE
            #Increment the time index
            tIdx += 1
        return eyePositions, rSimMat, P0_vect