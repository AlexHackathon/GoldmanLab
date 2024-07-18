import numpy as np
import matplotlib.pyplot as plt
import random

class Simulation:
    def __init__(self, neuronNum, dt, end, tau, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, nonLinearityFunction):
        self.neuronNum = neuronNum #Number of neurons in the simulation
        self.dt = dt #Time step [ms]
        self.t_end = end #Simulation end [ms]
        self.t_vect = np.arange(0, self.t_end, self.dt)

        self.eyeStart = eyeStartParam
        self.eyeStop = eyeStopParam
        self.eyeRes = eyeResParam
        self.maxFreq = maxFreq

        self.alpha = .1
        self.tau = tau
        self.f = nonLinearityFunction

        self.ksi = np.zeros((self.neuronNum))
        self.eta = np.zeros((self.neuronNum))

        self.onPoints = np.append(np.linspace(self.eyeStart, overlap, self.neuronNum//2), np.linspace(-overlap, self.eyeStop, self.neuronNum//2))
        self.cutoffIdx = np.zeros((self.neuronNum))
        self.eyePos = np.linspace(self.eyeStart, self.eyeStop, self.eyeRes)

        self.current_mat = np.zeros((self.neuronNum, len(self.t_vect))) #Defaults to no current
        self.w_mat = np.zeros((self.neuronNum,self.neuronNum))
        self.predictW = None
        self.s_mat = np.zeros((len(self.t_vect), self.neuronNum))
        self.T = np.zeros((self.neuronNum,))
        self.r_mat = self.CreateTargetCurves()

    def SetCurrent(self, currentMat):
        self.current_mat = currentMat
    def SetWeightMatrix(self, weightMatrix):
        self.w_mat = weightMatrix
    def CreateTargetCurves(self):
        slope = self.maxFreq / (self.eyeStop - self.eyeStart)
        r_mat = np.zeros((self.neuronNum, len(self.eyePos)))
        for n in range(self.neuronNum):
            for eIdx in range(len(self.eyePos)):
                #If neurons have positive slope and have 0s at the start
                y = None
                self.ksi[n] = slope
                if n < self.neuronNum//2:
                    y = slope * (self.eyePos[eIdx] - self.onPoints[n])
                    # Correction for negative numbers
                    if y < 0:
                        y = 0
                #If neurons have negative slope and end with 0s
                else:
                    self.ksi[n] = -slope
                    y = -slope * (self.eyePos[eIdx] - self.onPoints[n])
                    if y < 0:
                        y = 0
                r_mat[n][eIdx] = y
            #Set the tonic input to the non-corrected y intercept
            if n < self.neuronNum // 2:
                #self.T[n] = slope * (self.eyeStart - self.onPoints[n]) #Intercept with x=-25
                self.T[n] = slope * -self.onPoints[n]
            else:
                #self.T[n] = -slope * (self.eyeStop - self.onPoints[n]) + 150 #Intercept with x=25
                #self.T[n] = -slope * (- self.onPoints[n]) #Intercept with x=0
                self.T[n] = -slope * -self.onPoints[n]
        self.r_mat = r_mat

    def PlotTargetCurves(self):
        for r in range(len(self.r_mat)):
            plt.plot(self.eyePos, self.r_mat[r])
        plt.xlabel("Eye Position")
        plt.ylabel("Firing Rate")
        plt.show()
    def FitWeightMatrixProduct(self):
        F_mat = np.zeros((len(self.eyePos), self.neuronNum))
        for e in range(len(self.eyePos)):
            #funcIn = self.ksi * self.eyePos[e] + self.T #nx1 vector
            #func = Geometric(funcIn, .4, 1.4) #Creates an exn matrix
            func  = self.f(self.r_mat[:,e])
            F_mat[e] = func
        self.eta = np.linalg.lstsq(F_mat, self.eyePos)[0]
        self.w_mat = np.outer(self.ksi, self.eta)
    def PredictEyePosNonlinearSaturation(self, s_E):
        #Change: potentially include a tonic input in the prediction
        return np.dot(s_E, self.eta)
    def GetR(self, s_e):
        r_e = np.dot(self.w_mat, s_e) + self.T
        return r_e
    def GraphWeightMatrix(self):
        plt.imshow(self.w_mat)
        plt.title("Weight Matrix")
        plt.colorbar()
        plt.show()
    def Mistune(self, error):
        self.w_mat = (1-error) * self.w_mat
    def RunSim(self, startIdx=-1, plot=True, dead=[]):
        if not startIdx == -1:
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        else:
            # Set it to some default value in the middle
            startIdx = self.neuronNum // 2
            self.s_mat[0] = self.f(self.r_mat[:, startIdx])
        tIdx = 1
        eyePositions = np.zeros((len(self.t_vect)))
        eyePositions[0] = self.PredictEyePosNonlinearSaturation(self.s_mat[0])
        while tIdx < len(self.t_vect):
            current = np.zeros((self.neuronNum))
            mag = 0
            soa = 400
            dur = 100
            if tIdx%(soa/self.dt) >= 0 and tIdx%(soa/self.dt) < (dur/self.dt):
                for n in range(self.neuronNum):
                    if n < self.neuronNum//2:
                        current[n] = mag
                    else:
                        current[n] = -mag
            r_vect = np.array(np.dot(self.w_mat, self.s_mat[tIdx - 1]) + self.T + current)
            r_vect = [0 if r < 0 else r for r in r_vect] #CHANGE
            for d in dead:
                r_vect[d] = 0
            decay = -self.s_mat[tIdx - 1]
            growth = self.f(r_vect)
            self.s_mat[tIdx] = self.s_mat[tIdx-1] + self.dt/self.tau*(decay + growth)
            eyePositions[tIdx] = self.PredictEyePosNonlinearSaturation(self.s_mat[tIdx])
            tIdx += 1
        if plot:
            plt.plot(self.t_vect, eyePositions)
        plt.xlabel("Time (ms)")
        plt.ylabel("Eye Position (degrees)")
        return (abs(eyePositions[-1] - eyePositions[0]) <= 3)
    def PlotFixedPointsOverEyePosRate(self,neuronArray):
        for n in neuronArray:
            r = np.zeros(len(self.eyePos))
            for e in range(len(self.eyePos)):
                r[e] = np.dot(self.w_mat[n], self.f(self.r_mat[:,e])) + self.T[n]
                r[e] = max(0, r[e])
            plt.plot(self.eyePos, self.r_mat[n])
            plt.plot(self.eyePos, r)
            plt.xlabel("Eye Position")
            plt.ylabel("Fixed Points")

overlap = 5
neurons = 50
#(self, neuronNum, dt, end, tau, a, p, maxFreq, eyeStartParam, eyeStopParam, eyeResParam, nonlinearityFunction):
#Instantiate the simulation
alpha = .5
myNonlinearity = lambda r_vect: SynapticSaturation(r_vect, alpha)
#myNonlinearity = lambda r_vect: alpha * Geometric(r_vect, .4, 1.4)
sim = Simulation(neurons, .1, 1000, 20, 150, -25, 25, 1000, myNonlinearity)
#Create and plot the curves
sim.CreateTargetCurves()
sim.PlotTargetCurves()
sim.FitWeightMatrixProduct()
#Mistuning Code
"""for n in range(0,10):
    print(.01*n)
    sim.FitWeightMatrixProduct()
    sim.Mistune(.01*n)
    sim.PlotFixedPointsOverEyePosRate(range(neurons))
    plt.show()
    #Running simulations for different eye positions
    for e in range(len(sim.eyePos)):
        if e%100 == 0:
            sim.RunSim(startIdx=e)
    plt.show()"""
#Lesioning Code
for e in range(len(sim.eyePos)):
    if e%100 == 0:
        print(e)
        sim.RunSim(startIdx=e, dead=[12])
plt.show()

for e in range(len(sim.eyePos)):
    if e%100 == 0:
        print(e)
        sim.RunSim(startIdx=e, dead=[27])
plt.show()

#Running simulations for different eye positions
for e in range(len(sim.eyePos)):
    if e%100 == 0:
        print(sim.RunSim(startIdx=e))
plt.show()

#Running simulations for different eye positions (Lesioned)
numLesion = 1
for e in range(len(sim.eyePos)):
    if e%20 == 0:
        sim.RunSim(startIdx=e, dead=[random.randint(0,neurons-1) for i in range(numLesion)])
plt.show()