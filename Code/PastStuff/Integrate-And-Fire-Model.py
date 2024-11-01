#import numpy as np
import matplotlib as matplot

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

#Define vector for time
t_vect = [t_StimStart]
while t_vect[-1] < t_StimEnd:
    t_vect.append(t_vect[-1] + dt)
print(t_vect)

#Define vector for voltage and injected current
V_vect = [0] * len(t_vect)
V_vect[0] = E_L #set the start to the resting potential

I_e_vect = [0] * len(t_vect)

idx = 1

for i in t_vect:
    V_inf = E_L + I_e_vect[idx] * R_m
    V_vect[idx+1] = V_inf + (V_vect[idx] - V_inf) ** (-dt/tau) #plug into integrated formula
    idx = idx+1 #increment