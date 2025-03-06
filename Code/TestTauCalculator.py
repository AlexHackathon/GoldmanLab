import numpy as np
import matplotlib.pyplot as plt
import  scipy as sp
from scipy.optimize import curve_fit
from skopt import gp_minimize

def CalculateTau(x,y):
    #ATTENTION: five times the time constant produces a 3.5% underestimation error consistently,
    #Subtract min from all
    y = y - y[-1]
    #Scale so that the first value is one
    y = y/y[0]
    tau = sp.integrate.trapezoid(y,x)
    return tau
def DimensionGenerator(fitVect):
    A = fitVect[0]
    B = fitVect[1]
    t1 = fitVect[2]
    t2 = fitVect[3]
    aRange = (0.0,100.0)
    bRange = (0.0,100.0)
    t1Range = (0.0,100.0)
    t2Range = (100.0, 2000.0)
    dimensions = []
    if A==None:
        dimensions.append(aRange)
    else:
        dimensions.append((A,A+.0001))
    if B==None:
        dimensions.append(bRange)
    else:
        dimensions.append((B,B+.0001))
    if t1==None:
        dimensions.append(t1Range)
    else:
        dimensions.append((t1,t1+.0001))
    if t2==None:
        dimensions.append(t2Range)
    else:
        dimensions.append((t2,t2+.0001))
    return dimensions
def FullFitFunc(input, timeVect, eyeVect):
    A = input[0]
    B = input[1]
    t1 = input[2]
    t2 = input[3]
    computedData = np.array([A * np.exp(-time/t1) + B * np.exp(-time/t2) for time in timeVect])
    differences = np.array(eyeVect) - computedData
    distance = np.linalg.norm(differences)
    return distance
def Minimizer(fitVect, timeVect, eyeVect):
    dimensions = DimensionGenerator(fitVect)
    newFunc = lambda input: FullFitFunc(input, timeVect, eyeVect)
    res = gp_minimize(func=newFunc,
                      dimensions=dimensions,
                      acq_func="EI",
                      n_calls=15,  # the number of evaluations of f (15)
                      n_initial_points=10,  # the number of random initialization points (5)
                      noise=0.1 ** 2,  # the noise level (optional)
                      random_state=1234)
    ans = res["x"]
    ans.append(eyeVect[-1])
    return res["x"]
def FullFitFunc2(time, A, B, t1, t2, C):
    #print(A,B,t1,t2)
    power1 = np.power(np.e,-time/t1)
    power2 = np.power(np.e,-time/t2)
    return  A * power1 + B * power2 + C
def FindDist(a, b):
    return np.sum(np.square(a-b))
def MyCurveFitter(timeVect, eyeVect, guess):
    myFunc = lambda time, A_1, B_1, t1_1, t2_1 : FullFitFunc2(time, A_1, B_1, t1_1, t2_1, eyeVect[-1])
    popt, pcov = curve_fit(myFunc, timeVect, eyeVect, p0=guess)
    return popt