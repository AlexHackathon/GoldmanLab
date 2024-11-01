import math
import random
import numpy as np
import matplotlib.pyplot as plt
import Helpers.ActivationFunction as ActivationFunction
import Helpers.GraphHelpers as MyGraphing
import Helpers.CurrentGenerator as MyCurrent

eyeStart = -25
eyeStop = 25
eyeRes = 100 #Make 500 for real simulation
heighestFreq = 150
neuronNum = 5

fixedA = 1000
fixedP = 1
def CreateTargetCurves():
    x = np.linspace(eyeStart, eyeStop, eyeRes)
    slope = heighestFreq / (eyeStop - eyeStart)
    onPoints = np.linspace(eyeStart, eyeStop, neuronNum + 1)[:-1]
    r_mat = np.zeros((neuronNum, len(x)))
    for i in range(neuronNum):
        for eIdx in range(len(x)):
            if x[eIdx] < onPoints[i]:
                r_mat[i][eIdx] = 0
            else:
                # Point-slope formula y-yi = m(x-xi)
                r_mat[i][eIdx] = slope * (x[eIdx] - onPoints[i])
    return x, r_mat

def PlotTargetCurves(rMatParam, eyeVectParam):
    for r in range(len(rMatParam)):
        plt.plot(eyeVectParam, rMatParam[r], color='blue')
    plt.xlabel("Eye Position")
    plt.ylabel("Firing Rate")
    plt.show()

curves = CreateTargetCurves()

PlotTargetCurves(curves[1], curves[0])

def GetWeightsReverseEyePosNonlinear():
    S_mat = np.zeros(np.shape(curves[1]))
    S_mat = ActivationFunction.Geometric(curves[1], fixedA, fixedP) #Pass the target values through the activation function
    add = np.array(np.ones(len(S_mat[-1]))) #Creates an array of eyePosition number of 1s
    add = np.resize(add, (1, eyeRes)) #Reshapes that array into a 1 x eyePosition array
    S_tilda = np.append(S_mat, add, axis=0) #Creates a new activation function matrix with an extra row of 1s
    sTildaTranspose = np.transpose(S_tilda)  # Shape: (100,6)
    eyePos = curves[0]  # Shape: (50,)
    weightSolution = np.linalg.lstsq(sTildaTranspose, eyePos, rcond=None)[0]
    return weightSolution
def GetEyePosition(r_E, targetCurves):
    add = np.array(np.ones(len(targetCurves[1][-1])))  # Creates an array of eyePosition number of 1s
    add = np.resize(add, (1, len(add)))  # Reshapes that array into a 1 x eyePosition array
    r_tilda = np.append(targetCurves[1], add, axis=0)  # Creates a new activation function matrix with an extra row of 1s
    rTildaTranspose = np.transpose(r_tilda)  # Shape: (100,6)
    eyePos = curves[0]  # Shape: (50,)
    weightSolution = np.linalg.lstsq(rTildaTranspose, eyePos, rcond=None)[0]

    #Use the weights to calculate eye position
    t = weightSolution[-1]
    w = weightSolution[:-1]
    pos = np.dot(r_E, w) + t
    return pos