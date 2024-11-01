import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import math

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

#Running a simulation using a given nonlinear activation function on two recurrent neurons
#Constants for every run
weightMatrix = np.array([[0,25],[25,0]])
fixedA = 20
fixedP = 2
outputMatrix = np.zeros((2,len(t_vect)))
tIdx = 1
current1_vect = ConstCurrent(t_vect, 30, [t_stimStart,t_stimEnd])
current2_vect = ConstCurrent(t_vect, 30, [t_stimStart,t_stimEnd])


#Running the simulation for every weight in the weight values
while tIdx < len(t_vect):
    #Update rule for multiple connected nonlinear neurons    
    input_vect = outputMatrix[:,tIdx-1]
    #outputMatrix[:,tIdx] = outputMatrix[:,tIdx-1] + dt/tau*(-outputMatrix[:,tIdx-1] + np.dot(weightMatrix,Activation(fixedA,fixedP,input_vect)))
    outputMatrix[:,tIdx] = outputMatrix[:,tIdx-1] + dt/tau*(-outputMatrix[:,tIdx-1] + np.dot(weightMatrix,Activation(fixedA,fixedP,input_vect)) + [current1_vect[tIdx-1],current2_vect[tIdx-1]])
    tIdx = tIdx + 1
plt.plot(t_vect, outputMatrix[0])
plt.show()
#New added lines
skip = 100
r1 = np.zeros(int(len(t_vect)/skip))
r2 = np.zeros(int(len(t_vect)/skip))
c_vect = np.linspace(0,1,int(len(t_vect)/skip))
for t in range(0,int(len(t_vect)/skip)):
    r1[t] = outputMatrix[0,t*skip]
    r2[t] = outputMatrix[1,t*skip]
plt.scatter(r1,r2,c=c_vect,cmap='cool')
plt.colorbar()
plt.xlabel("Firing Rate Neuron 1")
plt.ylabel("Firing Rate Neuron 2")
#plt.plot(outputMatrix[0,t], outputMatrix[1,t],"ro", label=str(outputMatrix[0,t]) + "," + str(outputMatrix[1,t]),cmap="cool")
#plt.show()

#Plotting nullclines
w_12 = weightMatrix[0][1]
w_21 = weightMatrix[1][0]
I_1 = 0
I_2 = 0

r_vals = np.linspace(0,70, 70)
r1_nullcline = Activation(20,2,r_vals) * w_12 + I_1
r2_nullcline = Activation(20,2,r_vals) * w_21 + I_2

plt.plot(r_vals, r1_nullcline)
plt.plot(r2_nullcline, r_vals)

# Meshgrid
r1, r2 = np.meshgrid(np.linspace(0, 70, 20),  
                   np.linspace(0, 70, 20)) 
# Directional vectors
visC = 30
u = visC * (-r1 + Activation(20,2,r2) * w_12 + I_1)#1/tau * (-r1 + Activation(20,2,r2) * w_12 + I_1)
v = visC * (-r2 + Activation(20,2,r1) * w_21 + I_2)#1/tau * (-r2 + Activation(20,2,r1) * w_21 + I_2)
  
# Plotting Vector Field with QUIVER 
plt.quiver(r1, r2, u, v, color='g') 
plt.title('Nullclines for Two Recurrent Neurons 2 Fixed Points') 

  
# Show plot with grid 
plt.grid() 
plt.show()


#Running the simulation for 3 fixed points
#Constants for every run
weightMatrix = np.array([[0,25],[25,0]])
fixedA = 20
fixedP = 2
outputMatrix = np.zeros((2,len(t_vect)))
tIdx = 1
current1_vect = ConstCurrent(t_vect, 30, [t_stimStart,t_stimEnd])
current2_vect = ConstCurrent(t_vect, 30, [t_stimStart,t_stimEnd])
#Running the simulation for every weight in the weight values
while tIdx < len(t_vect):
    #Update rule for multiple connected nonlinear neurons    
    input_vect = outputMatrix[:,tIdx-1]
    outputMatrix[:,tIdx] = outputMatrix[:,tIdx-1] + dt/tau*(-outputMatrix[:,tIdx-1] + np.dot(weightMatrix,ActivationE(input_vect,r_m=10)) + [current1_vect[tIdx-1],current2_vect[tIdx-1]])
    tIdx = tIdx + 1
#New added lines
skip = 100
r1 = np.zeros(int(len(t_vect)/skip))
r2 = np.zeros(int(len(t_vect)/skip))
c_vect = np.linspace(0,1,int(len(t_vect)/skip))
for t in range(0,int(len(t_vect)/skip)):
    r1[t] = outputMatrix[0,t*skip]
    r2[t] = outputMatrix[1,t*skip]
plt.scatter(r1,r2,c=c_vect,cmap='cool')
plt.colorbar()
plt.xlabel("Firing Rate Neuron 1")
plt.ylabel("Firing Rate Neuron 2")
#Plotting nullclines 3 fixed points

#print(ActivationE(np.linspace(0,70,20)))
w_12 = weightMatrix[0][1]
w_21 = weightMatrix[1][0]
I_1 = 0
I_2 = 0
r_vals = np.linspace(0,70, 70)
r1_nullcline = ActivationE(r_vals,r_m=10) * w_12 + I_1
r2_nullcline = ActivationE(r_vals,r_m=10) * w_21 + I_2
plt.plot(r_vals, r1_nullcline)
plt.plot(r2_nullcline, r_vals)
# Meshgrid
r1, r2 = np.meshgrid(np.linspace(0, 70, 20),np.linspace(0, 70, 20))
# Directional vectors
visC = 30
u = visC*(-r1 + ActivationE(r2,r_m=10) * w_12 + I_1)
v = visC*(-r2 + ActivationE(r1,r_m=10) * w_21 + I_2)
# Plotting Vector Field with QUIVER
plt.quiver(r1, r2, u, v, color='g')
plt.title('Nullclines for Two Recurrent Neurons 3 Fixed Points') 
# Show plot with grid 
plt.grid() 
plt.show()
#Running the simulation for 1 fixed points
#Constants for every run
weightMatrix = np.array([[0,10],[10,0]])
fixedA = 20
fixedP = 2
outputMatrix = np.zeros((2,len(t_vect)))
tIdx = 1
current1_vect = ConstCurrent(t_vect, 30, [t_stimStart,t_stimEnd])
current2_vect = ConstCurrent(t_vect, 30, [t_stimStart,t_stimEnd])
#Running the simulation for every weight in the weight values
while tIdx < len(t_vect):
    #Update rule for multiple connected nonlinear neurons    
    input_vect = outputMatrix[:,tIdx-1]
    outputMatrix[:,tIdx] = outputMatrix[:,tIdx-1] + dt/tau*(-outputMatrix[:,tIdx-1] + np.dot(weightMatrix,ActivationE(input_vect,r_m=10)) + [current1_vect[tIdx-1],current2_vect[tIdx-1]])
    tIdx = tIdx + 1
#New added lines
skip = 100
r1 = np.zeros(int(len(t_vect)/skip))
r2 = np.zeros(int(len(t_vect)/skip))
c_vect = np.linspace(0,1,int(len(t_vect)/skip))
for t in range(0,int(len(t_vect)/skip)):
    r1[t] = outputMatrix[0,t*skip]
    r2[t] = outputMatrix[1,t*skip]
plt.scatter(r1,r2,c=c_vect,cmap='cool')
plt.colorbar()
plt.xlabel("Firing Rate Neuron 1")
plt.ylabel("Firing Rate Neuron 2")
#Plotting nullclines 3 fixed points

#print(ActivationE(np.linspace(0,70,20)))
w_12 = weightMatrix[0][1]
w_21 = weightMatrix[1][0]
I_1 = 0
I_2 = 0
r_vals = np.linspace(0,70, 70)
r1_nullcline = ActivationE(r_vals,r_m=10) * w_12 + I_1
r2_nullcline = ActivationE(r_vals,r_m=10) * w_21 + I_2
plt.plot(r_vals, r1_nullcline)
plt.plot(r2_nullcline, r_vals)
# Meshgrid
r1, r2 = np.meshgrid(np.linspace(0, 70, 20),np.linspace(0, 70, 20))
# Directional vectors
visC = 30
u = visC*(-r1 + ActivationE(r2,r_m=10) * w_12 + I_1)
v = visC*(-r2 + ActivationE(r1,r_m=10) * w_21 + I_2)
# Plotting Vector Field with QUIVER
plt.quiver(r1, r2, u, v, color='g')
plt.title('Nullclines for Two Recurrent Neurons 1 Fixed Points') 
# Show plot with grid 
plt.grid() 
plt.show()
