import time
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

dt = 0.1 #Time step [ms]
t_stimStart = 100 #Time start box current [ms]
t_stimEnd = 500 #Time stop box current [ms]
t_end = 1000 #Time to stop the simulation [ms]
tau = 20 #membrane time constant [ms]
t_vect = np.arange(0, t_end, dt) #Creates time vector[ms] with time step dt[ms]

def ConstCurrent(time_vect, stimMag, stimStartEnd_vect):
    addCurrent = False
    i = 0
    current_vect = np.zeros(len(time_vect))
    for x in range(0, len(time_vect)-1):
        if i >= len(stimStartEnd_vect):
            continue
        elif time_vect[x] >= stimStartEnd_vect[i]:
            addCurrent = not addCurrent
            i = i + 1
        if addCurrent:
            current_vect[x] = stimMag
    return current_vect
def Activation(a_param, p_param, r_param):
    numerator = r_param ** p_param
    denominator = a_param + r_param ** p_param
    return numerator/denominator
def ActivationE(r_input, r_m=0, width_param=1):
    #Default is the sigmoid with no change in width or center
    num = np.exp((r_input - r_m)/width_param)
    denom= 1 + num
    return num/denom
def Self(a_param, p_param, r_param):
    return r_param
def RunSim(weightMatrixParam, activationFunction, currentMatrix):
    tIdx = 1
    dimensions = (2, int(len(t_vect)))
    outputMatrix = np.zeros(dimensions)
    #Running the simulation for every weight in the weight values
    while tIdx < len(t_vect):
        #Update rule for multiple connected nonlinear neurons    
        input_vect = outputMatrix[:,tIdx-1]
        activationValues = activationFunction(input_vect)
        print(np.shape(weightMatrix))
        print(np.shape(activationValues))
        dotProd = np.dot(weightMatrix,activationValues)
        change = dt/tau*(-input_vect + dotProd + current[:,tIdx-1])
        outputMatrix[:,tIdx] = input_vect + change        
        tIdx = tIdx + 1
    return outputMatrix
def PlotSim(outputMatrix, sampleInterval, generatePlot=True):
    r1 = np.zeros(int(len(t_vect)/sampleInterval))
    r2 = np.zeros(int(len(t_vect)/sampleInterval))
    c_vect = np.linspace(0,1,int(len(t_vect)/sampleInterval))
    for t in range(0,int(len(t_vect)/sampleInterval)):
        r1[t] = outputMatrix[0,t*sampleInterval]
        r2[t] = outputMatrix[1,t*sampleInterval]
    plt.scatter(r1,r2,c=c_vect,cmap='cool')
    plt.colorbar()
    plt.xlabel("Firing Rate Neuron 1")
    plt.ylabel("Firing Rate Neuron 2")
    plt.plot()
def PlotNullcline(outputMatrix, weightMatrix, currentVect, activationFunction, gridDensity = 20, rMax=70, arrowScaling=30):
    w_12 = weightMatrix[0][1]
    w_21 = weightMatrix[1][0]
    I_1 = 0
    I_2 = 0

    r_vals = np.linspace(0,rMax, rMax)
    r1_nullcline = activationFunction() * w_12 + I_1
    r2_nullcline = activationFunction() * w_21 + I_2

    plt.plot(r_vals, r1_nullcline)
    plt.plot(r2_nullcline, r_vals)

    # Meshgrid
    r1, r2 = np.meshgrid(np.linspace(0, 70, 20),  
                       np.linspace(0, 70, 20)) 
    # Directional vectors
    u =  arrowScaling * (-r1 + activationFunction(r2) * w_12 + I_1)
    v =  arrowScaling * (-r2 + activationFunction(r1) * w_21 + I_2)
      
    # Plotting Vector Field with QUIVER 
    plt.quiver(r1, r2, u, v, color='g') 
    plt.title('Nullclines for Two Recurrent Neurons 2 Fixed Points')
    
    # Show plot with grid 
    plt.grid() 
    plt.show()

weightMatrix = np.array([[0,25],[25,0]])
def myActivationFunction1(input_vect):
    Activation(20,2,input_vect)
currentMatrix = [ConstCurrent(t_vect, 30, [t_stimStart,t_stimEnd]),
                 ConstCurrent(t_vect, 30, [t_stimStart,t_stimEnd])]
outputMatrix = RunSim(weightMatrix, myActivationFunction1, currentMatrix)
PlotSim(outputMatrix, 100, generatePlot=False)
PlotNullcline(outputMatrix, weightMatrix, myActivationFunction1)

