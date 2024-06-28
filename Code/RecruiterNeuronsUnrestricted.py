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
        self.t_end = end #Simulation end [ms]
        self.tau = tau
        self.t_vect = np.arange(0, self.t_end, self.dt)

        self.eyeStart = eyeStartParam
        self.eyeStop = eyeStopParam
        self.eyeRes = eyeResParam

        self.current_mat = np.zeros((self.neuronNum, len(self.t_vect))) #Defaults to no current
        #self.w_mat = np.zeros((self.neuronNum, self.neuronNum + 1)) #Tonic as last column
        self.w_mat = np.zeros((self.neuronNum,self.neuronNum))
        self.v_mat = np.zeros((len(self.t_vect), neuronNum)) #For simulation

        self.maxFreq = maxFreq
        self.onPoints = np.linspace(self.eyeStart, self.eyeStop, self.neuronNum)
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
            switch = False
            for eIdx in range(len(self.eyePos)):
                if self.eyePos[eIdx] < onPoints[i]:
                    r_mat[i][eIdx] = 0
                else:
                    # Point-slope formula y-yi = m(x-xi)
                    r_mat[i][eIdx] = slope * (self.eyePos[eIdx] - onPoints[i])
                    if not switch:
                        switch = True
                        self.onIdx[i] = eIdx
        return r_mat
    """def FitWeightMatrix2(self):
        self.r_mat = self.CreateTargetCurves()
        X = np.ones((len(self.eyePos), self.neuronNum + 1))
        Y = np.ones((len(self.eyePos), self.neuronNum + 1))
        for i in range(len(self.eyePos)):
            for j in range(self.neuronNum):
                X[i][j] = ActivationFunction.Geometric(self.r_mat[j][i], self.fixedA, self.fixedP)
                Y[i][j] = self.r_mat[j][i]
        for nIdx in range(self.neuronNum):
            r = self.r_mat[nIdx]
            solution = np.linalg.lstsq(X,r)[0]
            self.w_mat[nIdx] = solution[:-1]
            self.T[nIdx] = solution[-1]
            #self.T[nIdx] = min(0,solution[-1]) #Prevents negatives
        print()
        print()
        print(np.shape(self.w_mat))
        print(np.shape(self.T))
        self.FitPredictorNonlinear(Y)"""
    def FitWeightMatrixExclude2(self):
        self.r_mat = self.CreateTargetCurves()
        for n in range(self.neuronNum):
            startIdx = int(self.onIdx[n])
            #Do the fit
            X = np.ones((len(self.eyePos)-startIdx, self.neuronNum + 1))
            for i in range(len(X)):
                for j in range(len(X[0])-1):
                    X[i,j] = ActivationFunction.Geometric(self.r_mat[j,i+startIdx], self.fixedA, self.fixedP)
            print(np.shape(X))
            r = self.r_mat[n][startIdx:]
            solution = np.linalg.lstsq(X,r)[0]
            self.w_mat[n] =solution[:-1]
            self.T[n] = solution[-1]
        self.FitPredictorNonlinear()
    """def FitWeightMatrixExclude(self):
        self.r_mat = self.CreateTargetCurves()
        for n in range(self.neuronNum):
            startIdx = int(self.onIdx[n])
            #Do the fit
            X = np.ones((len(self.eyePos)-startIdx, self.neuronNum + 1))
            for i in range(len(X)):
                for j in range(len(X[i])-1):
                    X[i,j] = ActivationFunction.Geometric(self.r_mat[j][i+startIdx], self.fixedA, self.fixedP)
            r = self.r_mat[n][startIdx:]
            print(np.shape(r))
            print(np.shape(X))
            solution = np.linalg.lstsq(X,r)[0]
            self.w_mat[n] = solution[:-1]
            self.T[n] = solution[-1]
            if(n+1%10) == 0:
                quit()
        self.FitPredictorNonlinear()"""

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
    """def FitEyePosLinear(self):
        add = np.array(np.ones(len(self.r_mat[-1])))  # Creates an array of eyePosition number of 1s
        add = np.resize(add, (1, len(add)))  # Reshapes that array into a 1 x eyePosition array
        r_tilda = np.append(self.r_mat, add,
                            axis=0)  # Creates a new activation function matrix with an extra row of 1s
        rTildaTranspose = np.transpose(r_tilda)  # Shape: (100,6)
        weightSolution = np.linalg.lstsq(rTildaTranspose, self.eyePos, rcond=None)[0]
        self.predictW = weightSolution[:-1]
        self.predictT = weightSolution[-1]"""
    """def DumbPredictor(self, r_E):
        idx = np.searchsorted(self.r_mat[0], r_E[0], side="left")
        goal = idx
        if idx > 0 and (idx == len(self.r_mat[0]) or math.fabs(r_E[0] - self.r_mat[0][idx - 1]) < math.fabs(r_E[0] - self.r_mat[0][idx])):
            goal = idx-1
        return self.eyePos[goal]"""
    """def PredictEyePos(self, r_E):
        return np.dot(r_E, self.predictW) + self.predictT"""
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
    def PlotTargetCurves(self, rMatParam, eyeVectParam):
        colors = MyGraphing.getDistinctColors(self.neuronNum)
        for r in range(len(rMatParam)):
            #plt.plot(eyeVectParam, rMatParam[r], color = colors[r])
            plt.plot(eyeVectParam, rMatParam[r])
        plt.xlabel("Eye Position")
        plt.ylabel("Firing Rate")
        plt.show()
    def RunSim(self, startCond = np.empty(0), plot=False, color = None):
        #print("Running sim")
        if not np.array_equal(startCond, np.empty(0)):
            self.v_mat[0] = startCond
        else:
            self.v_mat[0] = sim.r_mat[:,0]
        tIdx = 1
        myFixed = []
        eyePositions = []
        while tIdx < len(self.t_vect):
            #Sets the basic values of the frame
            v = self.v_mat[tIdx-1]
            #print(np.shape(v))
            activation = ActivationFunction.Geometric(v,self.fixedA,self.fixedP)
            dot = np.dot(self.w_mat,activation)
            #print(np.shape(dot))
            delta = (-v + dot + self.T + self.current_mat[:,tIdx])
            a = -v + dot
            #print("A")
            #print(np.shape(a))
            b = a + self.T
            #print("B")
            #print(np.shape(b))
            #print(np.shape(self.T))
            c = b + self.current_mat[:,tIdx]
            #print("C")
            #print(np.shape(c))
            #print(delta)
            #print(np.shape(delta))
            #print(np.shape(self.v_mat[tIdx-1]))
            #print(np.shape(self.dt*self.tau*delta))
            self.v_mat[tIdx] = self.v_mat[tIdx-1] + self.dt/self.tau*delta
            for i in range(len(self.v_mat[tIdx])):
                self.v_mat[tIdx][i] = max(0,self.v_mat[tIdx][i])
            if len([d for d in delta if d <=.01]) == len(delta):
                add = True
                for p in myFixed:
                    if np.linalg.norm(p-self.v_mat[tIdx]) < 15:
                        add = False
                if add:
                    myFixed.append(self.v_mat[tIdx])
            #print(myFixed)
            self.fixedPoints.append(np.array(myFixed))
            if tIdx == 1:
                # Double adds the first eye position to correct for starting at 1
                #eyePositions.append(self.DumbPredictor(self.v_mat[tIdx]))
                #eyePositions.append(self.DumbPredictor(self.v_mat[tIdx]))
                eyePositions.append(self.PredictEyePosNonlinear(self.v_mat[tIdx]))
                eyePositions.append(self.PredictEyePosNonlinear(self.v_mat[tIdx]))
            else:
                eyePositions.append(self.PredictEyePosNonlinear(self.v_mat[tIdx]))
                #eyePositions.append(self.DumbPredictor(self.v_mat[tIdx]))
            tIdx += 1
        if plot:
            #Calculates the eye position at each time point in the simulation
            if color == None:
                plt.plot(self.t_vect, eyePositions)
            else:
                plt.plot(self.t_vect,eyePositions,color=color)
            #plt.show()
        if(len(eyePositions) < 2):
            print("GIANT ERROR")
            quit()
        return eyePositions[-1] - eyePositions[0]

    def GraphAllNeuronsTime(self):
        for nIdx in range(self.neuronNum):
            plt.plot(self.t_vect, self.v_mat[:, nIdx], label="Neuron " + str(nIdx))
        plt.xlabel("Time (ms)")
        plt.ylabel("Firing rate (spikes/sec)")
        plt.legend()
        plt.suptitle("Firing Rate Over Time")
        plt.show()

    """def GraphNeuronsTime(self):
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
            plt.show()"""
    def PlotFixedPointsOverEyePosAuto(self):
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
                    #print(p[n])
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
        y = []
        r = []
        for e in range(len(self.eyePos)):
            S = ActivationFunction.Geometric(self.r_mat[:,e], self.fixedA, self.fixedP)
            dot = np.dot(self.w_mat[nIdx],  S)
            y.append(dot + self.T[nIdx])
            r.append(self.r_mat[nIdx,e])
        plt.plot(myE,y)
        plt.plot(myE,r,color="blue")

#(self, neuronNum, dt, stimStart, stimEnd, end, tau, a, p, maxFreq):
neurons = 300
sim = Simulation(neurons, .1,1000, 20, .4, 1.4, 150, 0, 50, 200)
#sim.PlotTargetCurves(sim.r_mat, sim.eyePos)
sim.FitWeightMatrixNew()
#sim.FitWeightMatrixExclude2()
"""
sim.RunSim(plot=False, startCond=sim.r_mat[:,10])
plt.show()
sim.GraphAllNeuronsTime()
#sim.SetCurrent(np.array([MyCurrent.ConstCurrent(sim.t_vect,.0001,[500,110]) for n in range(neurons)]))
#sim.SetCurrent(MyCurrent.ConstCurrentBursts(sim.t_vect, 10, 10, 400, 0, 6000, neurons,sim.dt)) #10 and 100 because dt=0.1ms
#sim.SetCurrent(MyCurrent.ConstCurrentBursts(sim.t_vect, 200, 10000, 30000, 0, 10000, neurons))
#sim.PlotCurrent()
#Test the prediction neuron
for eIdx in range(len(sim.eyePos)):
    pos = sim.PredictEyePosNonlinear(sim.r_mat[:,eIdx])
    plt.scatter(sim.eyePos[eIdx],pos)
plt.show()
###IMPORTANT
for i in range(len(sim.r_mat[0])):
    if i%60==0:
        sim.RunSim(plot=True, startCond=sim.r_mat[:,i],color="Red")
plt.show()
'''
for e in range(neurons):
    sim.PlotFixedPointsOverEyePos(e)
plt.show()'''

#sim.FitWeightMatrix2()
'''
for eIdx in range(len(sim.eyePos)):
    pos = sim.PredictEyePosNonlinear(sim.r_mat[:,eIdx])
    plt.scatter(sim.eyePos[eIdx],pos)
    #print(sim.eyePos[eIdx])
    #print(sim.predictW)
plt.show()'''
'''
for i in range(len(sim.r_mat[0])):
    if i%3==0:
        sim.RunSim(plot=True, startCond=sim.r_mat[:,i])
plt.show()'''
'''
for e in range(neurons):
    sim.PlotFixedPointsOverEyePos(e)
plt.show()'''
#sim.PlotFixedPointsOverEyePos(0)
#plt.show()"""
#Try different simulation magnitudes to see which one is enough to integrate
"""I = np.linspace(0,1,10)
minAvg = None
minI = I[0]
print(minI)
for i in I:
    print(i)
    sim.SetCurrent(np.array([MyCurrent.ConstCurrent(sim.t_vect, i, [500, 110]) for n in range(neurons)]))
    avg = 0
    for i in range(len(sim.r_mat[0])):
        if i % 20 == 0:
            avg += sim.RunSim(plot=False, startCond=sim.r_mat[:, i])
    avg = avg/len(sim.r_mat[0])
    if minAvg == None:
        minAvg = avg/i
    else:
        if minAvg > avg/i:
            minAvg = avg/i
            minI = i
print(minI)"""
U,S,V = np.linalg.svd(sim.w_mat)
plt.plot(np.arange(1,len(S)+1,1), S)
plt.show()