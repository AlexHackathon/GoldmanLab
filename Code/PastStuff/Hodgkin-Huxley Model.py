import numpy as np
import matplotlib.pyplot as plt
import math
#********SETTING CONSTANT VALUES*********
dt = 0.1 #times step [ms]
t_end = 70 #end of the run[ms]
t_StimStart = 10 #start current injection [ms]
t_StimEnd = 60 #end current injection [ms]
c = 10 #capacitance per unit of area [nF/mm^2]
gmax_L = .003 * (10**3) #leak maximal conductance per unit area [uS/mm^2]
E_L = -54.387 #leak conductance reversal potential [mV]
gmax_K = .36 * (10**3) #hodkin-huxley maximal K conductance per unit area [uS/mm^2]
E_K = -77 #hodkin-huxley K conductance reversal potential [mV]
gmax_Na = 1.2 * (10**3) #hodkin-huxley maximal Na conductance per unit area [uS/mm^2]
E_Na = 50 #hodkin-huxley Na conductance reversal potential [mV]
#********INITIALIZING VARIABLES********
t_vect = np.arange(0, t_end, dt)
V_vect = np.zeros(len(t_vect))
m_vect = np.zeros(len(t_vect))
h_vect = np.zeros(len(t_vect))
n_vect = np.zeros(len(t_vect))
#********FORMULA FOR CALCULATING GATING VARIABLES********
'''Uses given formulas for alpha and beta in each case, as well as the general
formula x = alpha/(alpha+beta) to get the values for x_inf'''
def GetN(V):
    alpha = (.01*(V+ 55))/(1-math.exp(-.1*(V+55)))
    beta = .125 * math.exp(-.0125*(V+65))
    return alpha, beta
def GetM(V):
    alpha = (0.1*(V+40))/(1-math.exp(-.1*(V+40)))
    beta = 4 * math.exp(-.0556*(V+65))
    return alpha, beta
def GetH(V):
    alpha = .07 * math.exp(-.05*(V+65))
    beta = 1/(1 + math.exp(-.1*(V+35)))
    return alpha, beta
def GetTau(alpha_beta):
    return 1/(alpha_beta[0] + alpha_beta[1]) # 1/(alpha + beta)
def GetX_Inf(alpha_beta):
    return alpha_beta[0] / (alpha_beta[0] + alpha_beta[1]) #alpha/(alpha+beta)
#********FORMULA FOR CREATING CURRENT INJECTION VECTOR********
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
#********SETTING INITIAL VALUES********
i = 0
V_vect[i] = -65
m_vect[i] = GetX_Inf(GetM(V_vect[i]))
n_vect[i] = GetX_Inf(GetN(V_vect[i]))
h_vect[i] = GetX_Inf(GetH(V_vect[i]))
I_0 = 200 #magnitude of injected current [nA/mm^2]
I_e_vect = ConstCurrent(t_vect, I_0, [t_StimStart,t_StimEnd])
#********RUNNING THE SIMULATION********
for t in t_vect:
    if t == t_vect[-1]:
        break #prevents the i+1 breaking the code on the last index
    #Assign tau and x_inf for m,n,h
    tau_m = GetTau(GetM(V_vect[i]))
    m_inf = GetX_Inf(GetM(V_vect[i]))
    tau_h = GetTau(GetH(V_vect[i]))
    h_inf = GetX_Inf(GetH(V_vect[i]))
    tau_n = GetTau(GetN(V_vect[i]))
    n_inf = GetX_Inf(GetN(V_vect[i]))
    #Assign tau_V and V_inf
    V_denom = gmax_L + gmax_K*(n_vect[i]**4) + gmax_Na*(m_vect[i]**3)*h_vect[i]
    tau_V = c/V_denom
    V_inf = (gmax_L*E_L + gmax_K*(n_vect[i]**4)*E_K + gmax_Na*(m_vect[i]**3)*h_vect[i]*E_Na + I_e_vect[i])/V_denom
    #Assign next elements of m,h,n vectors using the update rule
    m_vect[i+1] = m_inf + (m_vect[i] - m_inf)*math.exp(-dt/tau_m)
    h_vect[i+1] = h_inf + (h_vect[i] - h_inf)*math.exp(-dt/tau_h)
    n_vect[i+1] = n_inf + (n_vect[i] - n_inf)*math.exp(-dt/tau_n)
    V_vect[i+1] = V_inf + (V_vect[i] - V_inf) * math.exp(-dt/tau_V)
    i = i+1
    
#********PLOTTING CODE FOLLOWS********
#Current
plt.plot(t_vect, I_e_vect)
plt.show()
#Voltage
fig, axs = plt.subplots(4)
fig.suptitle("Hodkin-Huxley Variables Over Time")
axs[0].plot(t_vect, V_vect)
axs[0].set_title("Voltage vs Time Graph")
axs[0].set_ylabel("Voltage (mV)")
#m
axs[1].set_ylabel("g_{Na} activation variable m")
axs[1].plot(t_vect, m_vect)
axs[1].set_title("g_{Na} activation variable m vs time")
#h
axs[2].set_ylabel("g_{Na} inactivation variable h")
axs[2].plot(t_vect, h_vect)
axs[2].set_title("g_{Na} inactivation variable h vs time")
#n
axs[3].set_ylabel("g_{K} activation variable n")
axs[3].plot(t_vect, n_vect)
axs[3].set_title("g_{K} activation variable n vs time")
plt.show()
