import random
import numpy
import numpy as np
import matplotlib.pyplot as plt
import math
import Helpers.ActivationFunction as ActivationFunction
import Helpers.GraphHelpers as MyGraphing
import Helpers.CurrentGenerator as MyCurrent


class Simulation:
    def __init__(self, neuronNum, dt, stimStart, stimEnd, end, tau, a, p, maxFreq, eyeStartParam, eyeStopParam,
                 eyeResParam):
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
        self.neuronNum = neuronNum  # Number of neurons in the simulation
        self.dt = dt  # Time step [ms]
        self.t_stimStart = stimStart  # Stimulation start [ms]
        self.t_stimEnd = stimEnd  # Stimulation end [ms]
        self.t_end = end  # Simulation end [ms]
        self.tau = tau
        self.t_vect = np.arange(0, self.t_end, self.dt)

        self.eyeStart = eyeStartParam
        self.eyeStop = eyeStopParam
        self.eyeRes = eyeResParam

        self.current_mat = np.zeros((self.neuronNum, len(self.t_vect)))  # Defaults to no current
        # self.w_mat = np.zeros((self.neuronNum, self.neuronNum + 1)) #Tonic as last column
        self.w_mat = np.zeros((self.neuronNum, self.neuronNum))
        self.eida = None
        self.ksi = None
        self.v_mat = np.zeros((len(self.t_vect), neuronNum))  # For simulation

        self.maxFreq = maxFreq
        self.onPoints = np.linspace(self.eyeStart, self.eyeStop, self.neuronNum + 1)[:-1]
        self.eyePos = np.linspace(self.eyeStart, self.eyeStop, self.eyeRes)
        self.r_mat = None
        self.T = -(self.maxFreq / (self.eyeStop - self.eyeStart))*self.onPoints

        self.fixedA = a
        self.fixedP = p
        # Fixed points is a list of the same length as eyePos
        # Each index contains a list of fixed points for that simulation
        # Each fixed point contains the values for each of the neurons at that time point
        self.fixedPoints = []  # NOT A SQUARE MATRIX
    def SetCurrent(self, currentMat):
        self.current_mat = currentMat
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
        self.r_mat = r_mat
    def TempAct(self, val, a, p):
        return val**p/(a+val**p)
    def FitColumn(self):
        S_mat = np.transpose(self.r_mat)
        print(np.shape(S_mat))
        for i in range(len(S_mat)):
            for j in range(len(S_mat[i])):
                S_mat[i,j] = self.TempAct(S_mat[i,j], self.fixedA, self.fixedP)
        self.eida = np.linalg.lstsq(S_mat, self.eyePos)[0]
        self.ksi = 3*np.ones(self.neuronNum)
        self.ksi = np.reshape(self.ksi, (len(self.ksi),1))
        print(np.shape(self.eida))
        print(np.shape(self.ksi))
        self.w_mat = np.outer(self.ksi, self.eida)
    def PredictEyePosNonlinear(self, r_E):
        return np.dot(self.eida, ActivationFunction.Geometric(r_E, self.fixedA, self.fixedP))

    def PlotTargetCurves(self, rMatParam, eyeVectParam):
        colors = MyGraphing.getDistinctColors(self.neuronNum)
        for r in range(len(rMatParam)):
            # plt.plot(eyeVectParam, rMatParam[r], color = colors[r])
            plt.plot(eyeVectParam, rMatParam[r])
        plt.xlabel("Eye Position")
        plt.ylabel("Firing Rate")
    def RunSim(self, startCond=np.empty(0), plot=False):
        # print("Running sim")
        if not np.array_equal(startCond, np.empty(0)):
            self.v_mat[0] = startCond
        else:
            self.v_mat[0] = sim.r_mat[:, 0]
        tIdx = 1
        eyePositions = []
        while tIdx < len(self.t_vect):
            # Sets the basic values of the frame
            v = self.v_mat[tIdx - 1]
            activation = ActivationFunction.Geometric(v, self.fixedA, self.fixedP)
            dot = np.dot(self.w_mat, activation)
            delta = (-v + dot + self.T + self.current_mat[:, tIdx])
            self.v_mat[tIdx] = self.v_mat[tIdx - 1] + self.dt / self.tau * delta
            for i in range(len(self.v_mat[tIdx])):
                self.v_mat[tIdx][i] = max(0, self.v_mat[tIdx][i])
            if plot:
                if tIdx == 1:
                    # Double adds the first eye position to correct for starting at 1
                    eyePositions.append(self.PredictEyePosNonlinear(self.v_mat[tIdx]))
                    eyePositions.append(self.PredictEyePosNonlinear(self.v_mat[tIdx]))
                else:
                    eyePositions.append(self.PredictEyePosNonlinear(self.v_mat[tIdx]))
            tIdx += 1
        if plot:
            # Calculates the eye position at each time point in the simulation
            plt.plot(self.t_vect, eyePositions)
            # plt.show()

    def GraphAllNeuronsTime(self):
        for nIdx in range(self.neuronNum):
            plt.plot(self.t_vect, self.v_mat[:, nIdx], label="Neuron " + str(nIdx))
        plt.xlabel("Time (ms)")
        plt.ylabel("Firing rate (spikes/sec)")
        plt.legend()
        plt.suptitle("Firing Rate Over Time")
        plt.show()

    def PlotFixedPointsOverEyePos(self):
        print("Plotting")
        colors = MyGraphing.getDistinctColors(self.neuronNum)  # WONT WORK FOR ANY NUMBER OF NEURONS (ONLY 1-5)
        for i in range(len(self.r_mat[0])):
            start = self.r_mat[:, i]
            start = start + [30 * (random.random() - .5) for d in range(len(start))]
            # self.RunSim(i, startCond=self.r_mat[:,i])
            # print(self.r_mat[:,i])
            if i == len(self.r_mat[0]) - 1:
                self.RunSim(startCond=self.r_mat[:, i], plot=True)
            else:
                self.RunSim(startCond=self.r_mat[:, i])
        print("Ran sims")
        for e in range(len(self.eyePos)):
            for p in self.fixedPoints[e]:
                for n in range(len(p)):
                    # print(colors[n])
                    # plt.scatter(self.eyePos[e], p[n], c=colors[n], s=4)
                    # print(p[n])
                    plt.scatter(self.eyePos[e], p[n], c='red', s=9)
            # plt.plot(self.eyePos, self.fixedPoints[j], label="Neuron "+str(j))
        print("Finished plots")
        # plt.xlabel("Eye Position")
        # plt.ylabel("Firing rate")
        plt.suptitle("Fixed Points Over Eye Position")
        self.PlotTargetCurves(self.r_mat, self.eyePos)
        plt.show()
    def PlotFixedPointsOverEyePos2(self, nIdx):
        y = []
        self.PlotTargetCurves(self.r_mat, self.eyePos)
        for e in range(len(self.eyePos)):
            interior = self.ksi * self.eyePos[e] + self.T
            ans = np.dot(self.eida, ActivationFunction.Geometric(interior, self.fixedA, self.fixedP))
            y.append(ans)
        plt.plot(self.eyePos, y)
        plt.plot(self.eyePos, self.eyePos)
        #plt.plot(myE, r, color="blue")


# (self, neuronNum, dt, stimStart, stimEnd, end, tau, a, p, maxFreq):
neurons = 100
sim = Simulation(neurons, .1, 100, 500, 1000, 20, .4, 1.4, 150, 0, 50, 200)
sim.CreateTargetCurves()
sim.PlotTargetCurves(sim.r_mat, sim.eyePos)
plt.show()
sim.FitColumn()
sim.CreateTargetCurves()
#print(sim.w_mat)
#print(sim.T)
# sim.SetCurrent(MyCurrent.ConstCurrentBursts(sim.t_vect, 200, 100, 300, 0, 6000, neurons)) #10 and 100 because dt=0.1ms
# Test the prediction neuron
for eIdx in range(len(sim.eyePos)):
    pos = sim.PredictEyePosNonlinear(sim.r_mat[:, eIdx])
    plt.scatter(sim.eyePos[eIdx], pos)
plt.show()
"""for eIdx in range(len(sim.eyePos)):
    if eIdx % 10 == 0:
        sim.RunSim(plot=True, startCond=sim.r_mat[:, eIdx])
plt.show()"""
sim.SetCurrent(MyCurrent.ConstCurrentBursts(sim.t_vect, 2, 100, 300, 0, 6000, neurons, sim.dt)) #10 and 100 because dt=0.1ms
for eIdx in range(len(sim.eyePos)):
    if eIdx % 10 == 0:
        sim.RunSim(plot=True, startCond=sim.r_mat[:, eIdx])
plt.show()
"""for e in range(neurons):
    sim.PlotFixedPointsOverEyePos2(e)
plt.show()"""