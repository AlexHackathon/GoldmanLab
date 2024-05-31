import numpy
import numpy as np
import matplotlib.pyplot as plt
import math
import Helpers.ActivationFunction as ActivationFunction
import Helpers.GraphHelpers as MyGraphing
import Helpers.CurrentGenerator as MyCurrent
class Simulation:
    def __init__(self, neuronNum, dt, stimStart, stimEnd, end, tau, a, p, t):
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

        self.current_mat = None
        self.w_mat = np.zeros((self.neuronNum, self.neuronNum))
        self.v_mat = np.zeros((len(self.t_vect), neuronNum)) #For simulation
        self.r_mat = None #NOT PERMANENT (The target values for the simulation)
        self.T = t

        self.fixedA = a
        self.fixedP = p
        self.onPoints = t #NOT PERMANENT

        self.lastDelta = None

        self.eyePos = np.linspace(0,100,20)
        #Fixed points is a list of the same length as eyePos
        #Each index contains a list of fixed points for that simulation
        #Each fixed point contains the values for each of the neurons at that time point
        self.fixedPoints = [] #NOT A SQUARE MATRIX
    def SetCurrent(self, I_e=None, mag = 0):
        if I_e != None:
            self.current_mat = I_e
        elif mag != 0:
            print("Setting the current by magnitude")
            self.current_mat = np.array([MyCurrent.ConstCurrent(self.t_vect, mag, [self.t_stimStart, self.t_stimEnd]) for n in range(self.neuronNum)])
        else:
            print("Didn't set the current in line 63")
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
    def FitWeightMatrix(self, slope=1):
        #Store firing rate in a matrix of firing rates over eye positions
        #Store the nonlinearities in a matrix of nonlinearities of each column over the eye position
        #Pseudoinverse moves the nonlinearity matrix to the other side of the equation
        #dr/dt and I are 0 because we are only looking at fixed points
        #T is set manually and equally spaced
        self.r_mat = np.zeros((self.neuronNum, len(self.eyePos)))
        for i in range(self.neuronNum):
            for eIdx in range(len(self.eyePos)):
                if self.eyePos[eIdx] < self.onPoints[i]:
                    self.r_mat[i][eIdx] = 0
                else:
                    #Point-slope formula y-yi = m(x-xi)
                    self.r_mat[i][eIdx] = slope * (self.eyePos[eIdx] - self.onPoints[i])
        self.PlotTargetCurves(self.r_mat, self.eyePos)
        #Setting S_mat (r x e)
        S_mat = np.zeros((self.neuronNum, len(self.eyePos)))
        for i in range(len(self.r_mat)):
            for j in range(len(self.r_mat[i])):
                S_mat[i][j] = ActivationFunction.Geometric(self.r_mat[i][j], self.fixedA, self.fixedP)
        #Setting T matrix (r x e)
        t_mat = np.zeros((len(self.T),len(self.eyePos)))
        for x in range(len(self.eyePos)):
            t_mat[:,x] = np.negative(self.T)
        #t_mat = [np.negative(self.T) for x in range(len(eyePos))] #Negative accounted for in this part
        self.w_mat =t_mat + self.r_mat #Should be -t_mat + r_mat in the math but did it in the previous step for simplicity
        inv = np.linalg.pinv(S_mat)
        self.w_mat = np.dot(self.w_mat, inv)
        print(self.w_mat)

    def PlotTargetCurves(self, rMatParam, eyeVectParam):
        colors = MyGraphing.getDistinctColors(self.neuronNum)
        for r in range(len(rMatParam)):
            plt.plot(eyeVectParam, rMatParam[r], color = colors[r])
        plt.xlabel("Eye Position")
        plt.ylabel("Firing Rate")
        #plt.show()

    def SetWeightMatrixRand(self, shift, scaling, seed):
        np.random.seed(seed)
        for i in range(len(self.w_mat)):
            for j in range(len(self.w_mat[i])):
                self.w_mat[i][j]=(np.random.randn()+ shift) * scaling
    def SetWeightMatrixManual(self, wMatParam):
        self.w_mat = wMatParam
    def RunSim(self, eyeIdx, startCond = np.empty(0)):
        #print("Running sim")
        if not np.array_equal(startCond, np.empty(0)):
            self.v_mat[0] = startCond
        tIdx = 1
        myFixed = []
        while tIdx < len(self.t_vect):
            #Sets the basic values of the frame
            v = self.v_mat[tIdx-1]
            #i = self.current_mat[:,tIdx-1]
            activation = ActivationFunction.Geometric(v,self.fixedA,self.fixedP)
            dot = np.dot(self.w_mat,activation)
            #delta = (-v + dot + i + self.T)
            delta = (-v + dot + self.T)
            self.v_mat[tIdx] = np.array([max(0,v) for v in self.v_mat[tIdx-1] + self.dt/self.tau*delta]) #Prevents negative
            #self.FindFixedPoints(self.dt/self.tau*delta)
            #if lastDelta != None and np.sign(self.dt/self.tau*delta) != lastDelta:
            #    fixedPoints.append(self.dt/self.tau*delta)
            #lastDelta = self.dt/self.tau*delta
            if len([d for d in delta if d <=.01]) == len(delta):
                add = True
                for p in myFixed:
                    if np.linalg.norm(p-self.v_mat[tIdx]) < 15:
                        add = False
                if add:
                    myFixed.append(self.v_mat[tIdx])
            #print(myFixed[-1])
            tIdx = tIdx + 1
        self.fixedPoints.append(np.array(myFixed))
    '''def RunSim(self, startCond = np.empty(0)):
        #print("Running sim")
        if not np.array_equal(startCond, np.empty(0)):
            self.v_mat[0] = startCond
        tIdx = 1
        myFixed = []
        while tIdx < len(self.t_vect):
            #Sets the basic values of the frame
            v = self.v_mat[tIdx-1]
            if tIdx % 50 == 0:
                i = np.array([10 for n in range(self.neuronNum)])
            else:
                i = np.array([0 for n in range(self.neuronNum)])
            #i = self.current_mat[:,tIdx-1]
            activation = ActivationFunction.Geometric(v,self.fixedA,self.fixedP)
            dot = np.dot(self.w_mat,activation)
            delta = (-v + dot + i + self.T)
            #delta = (-v + dot + self.T)
            self.v_mat[tIdx] = [max(0,v) for v in self.v_mat[tIdx-1] + self.dt/self.tau*delta] #Prevents negative
            #self.FindFixedPoints(self.dt/self.tau*delta)
            #if lastDelta != None and np.sign(self.dt/self.tau*delta) != lastDelta:
            #    fixedPoints.append(self.dt/self.tau*delta)
            #lastDelta = self.dt/self.tau*delta
            if len([d for d in delta if d <=.001]) == len(delta): #Are all neurons not changing
                add = True
                for p in myFixed:
                    if np.linalg.norm(p-self.v_mat[tIdx]) > 5:
                        add = False
                if add:
                    myFixed.append(self.v_mat[tIdx])
            tIdx = tIdx + 1
        self.fixedPoints.append(np.array(myFixed))'''
    def GraphNeuronsTime(self):
        while True:
            x1 = input("Number of neurons to graph: ")
            if x1=="break":
                break #Doesn't work
            try:
                x1 = int(x1)
            except:
                print("Not a valid neuron")
                continue
            for i in range(x1):
                nIdx=self.TakeNeuron()
                if nIdx == "break":
                    break
                plt.plot(self.t_vect, self.v_mat[:,nIdx], label="Neuron "+str(nIdx))
            plt.xlabel("Time (ms)")
            plt.ylabel("Firing rate (spikes/sec)")
            plt.legend()
            plt.suptitle("Firing Rate Over Time")
            plt.show()
    def PlotFixedPointsOverEyePos(self):
        print("Plotting")
        colors = MyGraphing.getDistinctColors(self.neuronNum) #WONT WORK FOR ANY NUMBER OF NEURONS (ONLY 1-5)
        for i in range(len(self.r_mat[0])):
            self.RunSim(i, startCond=self.r_mat[:,i])
            #self.GraphNeuronsTime()
        print("Ran sims")
        for e in range(len(self.eyePos)):
            for p in self.fixedPoints[e]:
                for n in range(len(p)):
                    #print(colors[n])
                    plt.scatter(self.eyePos[e], p[n], c=colors[n])
            #plt.plot(self.eyePos, self.fixedPoints[j], label="Neuron "+str(j))
        print("Finished plots")
        #plt.xlabel("Eye Position")
        #plt.ylabel("Firing rate")
        plt.show()
        #print("Finished")

#(self, neuronNum, dt, stimStart, stimEnd, end, tau, a, p, t):
neurons = 4
T = np.flip(np.linspace(20,100,neurons))
print(T)
sim = Simulation(neurons, .1, 100, 500, 1000, 20, .4, 1.4, T)
sim.FitWeightMatrix()
#sim.SetCurrent(mag=5)
sim.PlotFixedPointsOverEyePos()
#sim.GraphNeuronsTime()