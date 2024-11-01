import numpy as np
import matplotlib.pyplot as plt
import math

dt = 0.1 #time step [ms]
t_end = 500 #total time of run [ms]
t_StimStart = 100 #time to start injecting current [ms]
t_StimEnd = 400 #time to end injecting current [ms]
E_L = -70 #resting membrane potential [mV]
V_th = -55 # spike threshold [mV]
V_reset = -75 #value to rest the voltage to after a spike [mV]
V_spike = 20 # value to draw a spike to when the cell spikes[mV]
R_m = 10 #membrane resistance [MOhm]
tau = 10 #membrane time constant [ms]

#Function that takes a magnitude of current[nA] and times where the current is turned on and off
#Returns a vector 
def ConstCurrent(time_vect, stimMag, stimStartEnd_vect):
    addCurrent = False
    i = 0
    current_vect = np.zeros(len(time_vect))
    for x in range(0, len(time_vect)-1):
        if i >= len(stimStartEnd_vect):
            continue
        elif time_vect[x] >= stimStartEnd_vect[i]:s
            addCurrent = not addCurrent
            i = i + 1
        if addCurrent:
            current_vect[x] = stimMag
    return current_vect

#**************************************************************************************************
#Start of the running code not including defining starting parameters and functions
#Define vector for time from time 0 to time t_end
t_vect = np.arange(0, t_end, dt)

#Stimulated current declaration
I_Stim = np.arange(1.43, 1.67, 0.04)
switchTime_vect = [t_StimStart, t_StimEnd] #Times at which current is switched on or off

#Define what it means to run the simulation
def runSim(I_e_vect):
    idx = 0
    #Define vectors for the voltage of the cell over time (real and plotting)
    V_vect = np.zeros(len(t_vect))
    V_vect[0] = E_L #set the start to the resting potential
    V_plot_vect = np.zeros(len(t_vect))
    V_plot_vect[0] = V_vect[0]
    #Define a variable for counting the number of times the cell spiked
    numSpikes = 0
    while idx <= len(V_vect) - 2:
        V_inf = E_L + I_e_vect[idx] * R_m
        V_vect[idx+1] = V_inf + (V_vect[idx] - V_inf) * math.exp(-dt/tau) #plug into integrated formula
        if V_vect[idx+1] > V_th:
            V_vect[idx+1] = V_reset
            V_plot_vect[idx+1]= V_spike
            numSpikes = numSpikes + 1
        else:
            V_plot_vect[idx + 1] = V_vect[idx+1]
        idx = idx+1 #increment
    averageSpiking = 1000 * numSpikes / (t_StimEnd - t_StimStart)
    print("Average spikes: " + str(averageSpiking))
    return V_plot_vect

#Run the simulation for all values of I_Stim and plot current
#fig, axs = plt.subplots(len(I_Stim))
#fig.suptitle('Varrying I_Stim Current')
#for i in range(0,len(I_Stim)):
    #axs[i].plot(t_vect, ConstCurrent(t_vect, I_Stim[i], switchTime_vect))
    #plt.show()

#Run the simulation for all values of I_Stim and plot voltage
fig, axs = plt.subplots(len(I_Stim))
fig.suptitle('Varrying I_Stim Voltage')
for i in range(0,len(I_Stim)):
    axs[i].plot(t_vect, runSim(ConstCurrent(t_vect, I_Stim[i], switchTime_vect)))
plt.show()

#Plotting magnitude of current injected vs firing rate
I_threshold = (V_th - E_L)/R_m #current below which the cell does not fire
I_vect_long = np.arange(I_threshold + .001, 1.8, .001)
r_isi = []
for I in I_vect_long:
    numerator_log = (V_reset - E_L - I*R_m)
    denominator_log = (V_th - E_L - I* R_m)
    log_value = math.log(numerator_log/denominator_log)
    final_value = 1000/(tau * log_value)
    r_isi.append(final_value)
plt.plot(I_vect_long, r_isi)
plt.show()
