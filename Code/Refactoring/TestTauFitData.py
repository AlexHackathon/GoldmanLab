#region import statements
import numpy as np
import matplotlib.pyplot as plt
import pandas

import FacilitationSim as FacSim
import Refactoring.SimSupport as SimSupport
import Helpers.Bound as Bound
import SupplementalMaterialsGraphs as smg
import pickle
import pandas as pd
import TestTauCalculator as tc
import scipy.optimize as skopt
import os
import re
import time
#endregion
simTime, eyePositions = pickle.load(open("SimulationData/Data:E:14.28wf:0.5f:2e-05t_f:6000t_s50type:both", "rb"))
plt.plot(simTime, eyePositions)
plt.ylim([-20,20])
plt.show()

#region Define Double Exponential Fitter fit_double_exponential(x,y,initial)
def exponentialNormalization(xData, yData, killIdx):
    alignedXData = xData[killIdx:] - xData[killIdx]
    cutYData = yData[killIdx:]
    plt.plot(alignedXData, cutYData)
    plt.show()
    return alignedXData, cutYData
def double_exponential(x, a1, tau1, a2, tau2, offset):
    """Defines the double exponential function."""
    return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + offset

def fit_double_exponential(x_data, y_data):#initial_guess, startTime):
    """Fits a double exponential function using scipy.optimize.minimize."""
    popt, pcov = skopt.curve_fit(double_exponential, x_data, y_data)
    return popt

#Test data
normX, normY = exponentialNormalization(simTime, eyePositions, 10001)
plt.plot(normX, normY)
plt.show()

A, t_fast, B, t_slow, offset = fit_double_exponential(normX, normY)
yCalc = double_exponential(normX, A, t_fast, B, t_slow, offset)
plt.plot(simTime, eyePositions)
plt.plot(simTime[10001:], yCalc)
plt.show()
print((A, t_fast, B, t_slow, offset))