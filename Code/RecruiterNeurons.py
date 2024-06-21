import random
import numpy
import numpy as np
import matplotlib.pyplot as plt
import math
import Helpers.ActivationFunction as ActivationFunction
import Helpers.GraphHelpers as MyGraphing
import Helpers.CurrentGenerator as MyCurrent
class Simulation:
    def __init__(self, neuronNum, dt, stimStart, stimEnd, end, tau, a, p, maxFreq, eyeStartParam, eyeStopParam, eyeResParam):
        '''Instantiates the simulation
           ---------------------------
           Parameters
           neuronNum: the number of neurons simulated
           dt: the time step of the simulation in ms
           t_stimStart: the start of the electric current stimulation
           t_stimEnd: the end of the electric current stimulation
           t_end: the end of the simulation
           tau: the neuron time constant for all neurons
           currentMatrix: a matrix with each row representing the injected current of a neuron over time
           a: the a parameter of the system's nonlinearity
           p: the p parameter of the system's nonlinearity'''
        self.neuronNum = neuronNum #Number of neurons in the simulation
        self.dt = dt #Time step [ms]
        self.t_stimStart = stimStart #Stimulation start [ms]
        self.t_stimEnd = stimEnd #Stimulation end [ms]
        self.t_end = end #Simulation end [ms]
        self.tau = tau
        self.t_vect = np.arange(0, self.t_end, self.dt)

        self.current_mat = np.zeros((self.neuronNum, len(self.t_vect))) #Defaults to no current
        self.w_mat = np.zeros((self.neuronNum, self.neuronNum + 1)) #Tonic as last column
        self.v_mat = np.zeros((len(self.t_vect), neuronNum)) #For simulation
        self.eyeStart = eyeStartParam
        self.eyeStop = eyeStopParam
        self.eyeRes = eyeResParam
        self.r_mat = None #NOT PERMANENT (The target values for the simulation)
        self.T = None

        self.fixedA = a
        self.fixedP = p

        self.lastDelta = None

        self.onPoints = np.linspace(self.eyeStart, self.eyeStop, self.neuronNum)
        self.eyePos = np.linspace(self.eyeStart,self.eyeStop,self.eyeRes)
        #Fixed points is a list of the same length as eyePos
        #Each index contains a list of fixed points for that simulation
        #Each fixed point contains the values for each of the neurons at that time point
        self.fixedPoints = [] #NOT A SQUARE MATRIX
        self.maxFreq = maxFreq
    def SetCurrent(self, currentMat):
        self.current_mat = currentMat
    def PlotCurrent(self):
        for n in range(self.neuronNum):
            plt.plot(self.t_vect, self.current_mat[n])
        plt.suptitle("Current Magnitude Over Time")
        plt.xlabel("Time (ms)")
        plt.ylabel("Current Magnitude")
        plt.show()
    def TakeNeuron(self):
        x1Int = True
        x1 = None
        while x1Int:
            x1 = input("Neuron: ")
            if x1=="break":
                break
            if x1 == "quit":
                quit()
            try:
                x1 = int(x1)
                x1Int = False
            except:
                print("Not a valid neuron")
            if x1 >= self.neuronNum:
                print("Not a valid neuron")
                x1Int = True
        return x1

    def CreateTargetCurves(self):
        slope = self.maxFreq / (self.eyeStop - self.eyeStart)
        onPoints = np.linspace(self.eyeStart, self.eyeStop, self.neuronNum + 1)[:-1]
        r_mat = np.zeros((self.neuronNum, len(self.eyePos)))
        for i in range(self.neuronNum):
            for eIdx in range(len(self.eyePos)):
                if self.eyePos[eIdx] < onPoints[i]:
                    r_mat[i][eIdx] = 0
                else:
                    # Point-slope formula y-yi = m(x-xi)
                    r_mat[i][eIdx] = slope * (self.eyePos[eIdx] - onPoints[i])
        return r_mat
    def FitWeightMatrixNew(self, slope=1):
        #Store firing rate in a matrix of firing rates over eye positions
        #Use scipy.linalg.lsq_linear() to solve for the weight matrix row by row
        #dr/dt and I are 0 because we are only looking at fixed points
        self.r_mat = self.CreateTargetCurves()
        # Setting S_mat (n+1 x e)
        S_mat = ActivationFunction.Geometric(self.r_mat, self.fixedA, self.fixedP)
        print(np.shape(S_mat,))
        ones = np.array(np.ones(len(S_mat[-1])))
        ones = np.reshape(ones, (1,len(ones)))
        print(np.shape(ones))
        S_mat = np.append(S_mat, ones, axis=0)
        for k in range(len(self.w_mat)):
            #r_e and S~(r) transpose
            r = np.array(self.r_mat[k]) #Shape: (50,)
            sTildaTranspose = np.transpose(S_mat) #Shape: (50,6)
            weightSolution = np.linalg.lstsq(sTildaTranspose, r, rcond=None)[0]
            self.w_mat[k] = weightSolution
        self.T = self.w_mat[:,-1]
        self.w_mat = self.w_mat[:,0:len(self.w_mat[0])-1]
        print(np.shape(self.w_mat))
        print(self.T)
        print(self.w_mat)

    def PredictEyePos(self, r_E):
        add = np.array(np.ones(len(self.r_mat[-1])))  # Creates an array of eyePosition number of 1s
        add = np.resize(add, (1, len(add)))  # Reshapes that array into a 1 x eyePosition array
        r_tilda = np.append(self.r_mat, add,
                            axis=0)  # Creates a new activation function matrix with an extra row of 1s
        rTildaTranspose = np.transpose(r_tilda)  # Shape: (100,6)
        weightSolution = np.linalg.lstsq(rTildaTranspose, self.eyePos, rcond=None)[0]
        # Use the weights to calculate eye position
        t = weightSolution[-1]
        w = weightSolution[:-1]
        pos = np.dot(r_E, w) + t
        return pos

    def PredictEyePosNonlinear(self, r_E):
        S_mat = ActivationFunction.Geometric(self.r_mat, self.fixedA, self.fixedP)  # Pass the target values through the activation function
        S_mat = np.array(S_mat)
        add = np.array(np.ones(len(S_mat[-1])))  # Creates an array of eyePosition number of 1s
        add = np.resize(add, (1, len(add)))  # Reshapes that array into a 1 x eyePosition array
        S_tilda = np.append(S_mat, add,axis=0)  # Creates a new activation function matrix with an extra row of 1s
        sTildaTranspose = np.transpose(S_tilda)  # Shape: (100,6)
        weightSolution = np.linalg.lstsq(sTildaTranspose, self.eyePos, rcond=None)[0]

        t = weightSolution[-1]
        w = weightSolution[:-1]
        pos = np.dot(r_E, w) + t
        return pos
    def PlotTargetCurves(self, rMatParam, eyeVectParam):
        colors = MyGraphing.getDistinctColors(self.neuronNum)
        for r in range(len(rMatParam)):
            #plt.plot(eyeVectParam, rMatParam[r], color = colors[r])
            plt.plot(eyeVectParam, rMatParam[r], color='blue')
        plt.xlabel("Eye Position")
        plt.ylabel("Firing Rate")
        plt.show()
    def SetWeightMatrixRand(self, shift, scaling, seed):
        np.random.seed(seed)
        for i in range(len(self.w_mat)):
            for j in range(len(self.w_mat[i])):
                self.w_mat[i][j]=(np.random.randn()+ shift) * scaling
    def SetWeightMatrixManual(self, wMatParam):
        self.w_mat = wMatParam
    def PlotSteadyState(self):
        #Neuron 1 reference neuron
        x = np.linspace(0, self.maxFreq, self.eyeRes)
        print("Dimensions of X: " + str(np.shape(x)))
        # Plot r vs r
        y1 = np.linspace(0, self.maxFreq, self.eyeRes)  # r
        plt.plot(x,y1)
        for n in range(self.neuronNum):
            for e in range(len(self.r_mat)):
                #Plot r vs Steady State
                plt.scatter(self.r_mat[e][n], -self.r_mat[e][n])
                plt.scatter(self.r_mat[e][n], np.dot(self.w_mat[n], self.r_mat[:,e]) + self.T[n])
                #steadyState =
                #y2 = [steadyState for xPos in x]
                #Do the actual plotting
                #plt.plot(x,y2)
        plt.show()
    def RunSim(self, startCond = np.empty(0), plot=False):
        #print("Running sim")
        if not np.array_equal(startCond, np.empty(0)):
            self.v_mat[0] = startCond
        tIdx = 1
        myFixed = []
        eyePositions = []
        while tIdx < len(self.t_vect):
            #Sets the basic values of the frame
            v = self.v_mat[tIdx-1]
            activation = ActivationFunction.Geometric(v,self.fixedA,self.fixedP)
            dot = np.dot(self.w_mat,activation)
            delta = (-v + dot + self.T + self.current_mat[:,tIdx])
            self.v_mat[tIdx] = np.array([max(0,v) for v in self.v_mat[tIdx-1] + self.dt/self.tau*delta]) #Prevents negative
            if len([d for d in delta if d <=.01]) == len(delta):
                add = True
                for p in myFixed:
                    if np.linalg.norm(p-self.v_mat[tIdx]) < 15:
                        add = False
                if add:
                    myFixed.append(self.v_mat[tIdx])
            #print(myFixed)
            self.fixedPoints.append(np.array(myFixed))
            if plot:
                if tIdx == 1:
                    # Double adds the first eye position to correct for starting at 1
                    eyePositions.append(self.PredictEyePos(self.v_mat[tIdx]))
                    eyePositions.append(self.PredictEyePos(self.v_mat[tIdx]))
                else:
                    eyePositions.append(self.PredictEyePos(self.v_mat[tIdx]))
            tIdx += 1
        if plot:
            #Calculates the eye position at each time point in the simulation
            plt.plot(self.t_vect, eyePositions)
            plt.show()
    def GraphNeuronsTime(self):
        while True:
            x1 = input("Number of neurons to graph: ")
            if x1=="break":
                break #Doesn't work
            if not x1 == "all":
                try:
                    x1 = int(x1)
                except:
                    print("Not a valid neuron")
                    continue
            else:
                x1 = self.neuronNum
            for i in range(x1):
                nIdx = -1
                if x1 != self.neuronNum:
                    nIdx=self.TakeNeuron()
                    if nIdx == "break":
                        break
                else:
                    nIdx=i
                plt.plot(self.t_vect, self.v_mat[:,nIdx], label="Neuron "+str(nIdx))
                #if i == x1-1:
                #    print(self.v_mat[i])
            plt.xlabel("Time (ms)")
            plt.ylabel("Firing rate (spikes/sec)")
            plt.legend()
            plt.suptitle("Firing Rate Over Time")
            plt.show()
    def PlotFixedPointsOverEyePos(self):
        print("Plotting")
        colors = MyGraphing.getDistinctColors(self.neuronNum) #WONT WORK FOR ANY NUMBER OF NEURONS (ONLY 1-5)
        for i in range(len(self.r_mat[0])):
            start = self.r_mat[:,i]
            start = start + [30 * (random.random()-.5) for d in range(len(start))]
            #self.RunSim(i, startCond=self.r_mat[:,i])
            #print(self.r_mat[:,i])
            if i == len(self.r_mat[0])-1:
                self.RunSim(startCond=self.r_mat[:, i],plot=True)
            else:
                self.RunSim(startCond=self.r_mat[:, i])
        print("Ran sims")
        for e in range(len(self.eyePos)):
            for p in self.fixedPoints[e]:
                for n in range(len(p)):
                    #print(colors[n])
                    #plt.scatter(self.eyePos[e], p[n], c=colors[n], s=4)
                    print(p[n])
                    plt.scatter(self.eyePos[e], p[n], c='red', s=9)
            #plt.plot(self.eyePos, self.fixedPoints[j], label="Neuron "+str(j))
        print("Finished plots")
        #plt.xlabel("Eye Position")
        #plt.ylabel("Firing rate")
        plt.suptitle("Fixed Points Over Eye Position")
        self.PlotTargetCurves(self.r_mat, self.eyePos)
        plt.show()
    def PlotFixedPointsOverEyePos(self, nIdx):
        myE = np.linspace(self.eyeStart,self.eyeStop, self.eyeRes)
        print("X axis length: " + str(len(myE)))
        y = []
        r = []
        for e in range(len(self.eyePos)):
            y.append(np.dot(self.w_mat[nIdx], ActivationFunction.Geometric(self.r_mat[:,e], self.fixedA, self.fixedP) ) + self.T[nIdx])
            r.append(self.r_mat[nIdx,e])
        print(y)
        plt.plot(myE,y)
        plt.plot(myE,r)
        plt.show()



#(self, neuronNum, dt, stimStart, stimEnd, end, tau, a, p, maxFreq):
neurons = 20
sim = Simulation(neurons, .1, 100, 500, 1000, 20, 1, 1.4, 150, -25, 25, 100)
sim.FitWeightMatrixNew()
sim.PlotTargetCurves(sim.r_mat, sim.eyePos)
#sim.SetCurrent(MyCurrent.ConstCurrentBursts(sim.t_vect, 200, 100, 300, 0, 6000, neurons)) #10 and 100 because dt=0.1ms
#sim.SetCurrent(MyCurrent.ConstCurrentBursts(sim.t_vect, 200, 10000, 30000, 0, 10000, neurons))
#sim.PlotCurrent()
#sim.RunSim(plot=True)
#sim.GraphNeuronsTime()
#sim.PlotSteadyState()
sim.PlotFixedPointsOverEyePos(0)