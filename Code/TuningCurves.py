import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def OneSided():
    print("Sorry this isn't implemented right now")
def TwoSidedSameSlope(maxFreq, eyeStop, eyeStart, eyePosNum, neuronNum):
    print("Sorry this isn't implemented right now")
def TwoSidedDifferentSlope(fileLocation, eyeStart, eyeStop, eyePosNum):
        #"/Users/alex/Documents/Github/GoldmanLab/Code/EmreThresholdSlope_NatNeuroCells_All (1).xls"
        dataFrame = pd.read_excel(fileLocation)
        slopes = dataFrame["slope"]
        intercepts = dataFrame["thresh"]
        neuronNum = len(slopes)
        eyePos = np.linspace(eyeStart, eyeStop, eyePosNum)
        r_mat = np.zeros((neuronNum, eyePosNum))
        for eIdx in range(eyePosNum):
            r_e = np.multiply(slopes, np.subtract(np.ones(neuronNum) * eyePos[eIdx], intercepts))
            r_e = np.array([r if r > 0 else 0 for r in r_e])
            r_mat[:,eIdx] = r_e
        return r_mat
def TwoSidedDifferentSlopeMirror(fileLocation, eyeStart, eyeStop, eyePosNum):
    # "/Users/alex/Documents/Github/GoldmanLab/Code/EmreThresholdSlope_NatNeuroCells_All (1).xls"
    dataFrame = pd.read_excel(fileLocation)
    slopes = dataFrame["slope"]
    x_thresh = dataFrame["thresh"]
    neuronNumHalf = len(slopes)
    eyePos = np.linspace(eyeStart, eyeStop, eyePosNum)
    r_mat = np.zeros((neuronNumHalf * 2, eyePosNum))
    for eIdx in range(eyePosNum):
        r_e = np.multiply(slopes, np.subtract(np.ones(neuronNumHalf) * eyePos[eIdx], x_thresh))
        r_e = np.array([r if r > 0 else 0 for r in r_e])
        r_mat[:neuronNumHalf, eIdx] = r_e
        r_e2 = np.multiply(-slopes, np.subtract(np.ones(neuronNumHalf) * eyePos[eIdx], -x_thresh))
        r_e2 = np.array([r if r > 0 else 0 for r in r_e2])
        r_mat[neuronNumHalf:, eIdx] = r_e2
    return r_mat

def FindMin(fileLocation):
    minIdx = 0
    # "/Users/alex/Documents/Github/GoldmanLab/Code/EmreThresholdSlope_NatNeuroCells_All (1).xls"
    dataFrame = pd.read_excel(fileLocation)
    x_thresh = dataFrame["thresh"]
    for i in range(len(x_thresh)):
        if x_thresh[i] <= x_thresh[minIdx]:
            minIdx = i
    return minIdx

def TwoSidedDifferentSlopeMirrorNeg(fileLocation, eyeStart, eyeStop, eyePosNum):
    # "/Users/alex/Documents/Github/GoldmanLab/Code/EmreThresholdSlope_NatNeuroCells_All (1).xls"
    dataFrame = pd.read_excel(fileLocation)
    slopes = dataFrame["slope"]
    x_thresh = dataFrame["thresh"]
    neuronNumHalf = len(slopes)
    eyePos = np.linspace(eyeStart, eyeStop, eyePosNum)
    r_mat = np.zeros((neuronNumHalf * 2, eyePosNum))
    for eIdx in range(eyePosNum):
        r_e = np.multiply(slopes, np.subtract(np.ones(neuronNumHalf) * eyePos[eIdx], x_thresh))
        r_mat[:neuronNumHalf, eIdx] = r_e
        r_e2 = np.multiply(-slopes, np.subtract(np.ones(neuronNumHalf) * eyePos[eIdx], -x_thresh))
        r_mat[neuronNumHalf:, eIdx] = r_e2
    return r_mat
def PlotTuningCurves(r_mat, eyePos):
    '''Plot given target curves over eye position.'''
    for r in range(len(r_mat)):
        plt.plot(eyePos, r_mat[r])
    plt.xlabel("Eye Position")
    plt.ylabel("Firing Rate")
    plt.show()
def GetCutoffIdx(fileLocation, eyeStart, eyeStop, eyePosNum):
    dataFrame = pd.read_excel(fileLocation)
    eyePos = np.linspace(eyeStart, eyeStop, eyePosNum)
    x_thresh = dataFrame["thresh"]
    cutOffs = np.zeros((len(x_thresh),))
    for t in range(len(x_thresh)):
        difference = np.abs(eyePos - x_thresh[t])
        cutOffs[t] = np.argmin(difference)
    return cutOffs

def GetCutoffIdxMirror(fileLocation, eyeStart, eyeStop, eyePosNum):
    dataFrame = pd.read_excel(fileLocation)
    eyePos = np.linspace(eyeStart, eyeStop, eyePosNum)
    x_threshL = dataFrame["thresh"]
    x_thresh = np.append(x_threshL, -x_threshL)
    cutOffs = np.zeros((len(x_thresh),))
    for t in range(len(x_thresh)):
        difference = np.abs(eyePos - x_thresh[t])
        cutOffs[t] = np.argmin(difference)
    return cutOffs
