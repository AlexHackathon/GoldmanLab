import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as anim
import math

dt = 0.1 #Time step [ms]
t_stimStart = 0 #Time start box current [ms]
t_stimEnd = 500 #Time stop box current [ms]
t_end = 1000 #Time to stop the simulation [ms]
I_e = 1.43 #Magnitude of current [spikes/sec]

tau = 18 #membrane time constant [ms]
t_vect = np.arange(0, t_end, dt) #Creates time vector[ms] with time step dt[ms]

#Creates a box current over the time vector with magnitude stimMag
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


current1a_vect = ConstCurrent(t_vect, 63, [t_stimStart, t_stimEnd]) #Current injected into Neuron 1 [spikes/sec] 63
current1b_vect = ConstCurrent(t_vect, 57, [t_stimStart, t_stimEnd]) #Current injected into Neuron 2 [spikes/sec] 57
#Integrator (.3, -.7)
w_self = .2
w_other = -.7
weightMatrix = np.array([[w_self,w_other],
                        [w_other,w_self]]) #Adjustable weight matrix (-1 to 1 preferrable)
v1a_vect = np.zeros(len(t_vect)) #Output of Neuron 1 [spikes/sec]
v1b_vect = np.zeros(len(t_vect)) #Output of Neuron 2 [spikes/sec]
v1a_vect[0] = 63
v1b_vect[0] = 57
tIdx = 1 #Time index used for iteration

while tIdx < len(t_vect):
    #Calculating current in Neuron 1
    #u1a_vect = [0,v1b_vect[tIdx-1]] #Input of each neuron
    #totalInput1a = current1a_vect[tIdx-1] + sum(np.dot(u1a_vect, weightMatrix)) #Total input including decay and injected current
    #totalInput1a = sum(np.dot(u1a_vect, weightMatrix))
    #v1a_vect[tIdx] = v1a_vect[tIdx-1] + dt/tau*(-v1a_vect[tIdx-1] + totalInput1a) #Stepwise calculation of the next firing rate

    #Calculating current in Neuron 2
    #See previous comment
    #u1b_vect = [v1a_vect[tIdx-1],0] 
    #totalInput1b =  current1b_vect[tIdx-1] + sum(np.dot(u1b_vect, weightMatrix))
    #totalInput1b = sum(np.dot(u1b_vect, weightMatrix))
    #v1b_vect[tIdx] = v1b_vect[tIdx-1] + dt/tau*(-v1b_vect[tIdx-1] + totalInput1b)
    u = np.array([v1a_vect[tIdx-1],v1b_vect[tIdx-1]])
    i = np.array([current1a_vect[tIdx-1],current1b_vect[tIdx-1]])
    #if tIdx < 10:
    #    print("Input: " + str(u))
    v_vect_t = u + dt/tau*(-u + np.dot(u, weightMatrix) + i)
    #if tIdx < 10:
    #    print("Dot: " + str(np.dot(u,weightMatrix)))
    #    print("Sum: " + str(-u + np.dot(u,weightMatrix)))
    #    print("Change: " + str(dt/tau*(-u + np.dot(u, weightMatrix))))
    #    print("Final: " + str(v_vect_t))
    #if tIdx%100 == 0:
    #    print(v_vect_t)
    v1a_vect[tIdx] = v_vect_t[0]
    v1b_vect[tIdx] = v_vect_t[1]
    #Increase the time index
    tIdx = tIdx + 1
plt.plot(t_vect, v1a_vect, label="Neuron 1 (63Hz)")
plt.plot(t_vect, v1b_vect, label="Neuron 2 (57Hz)")
plt.xlabel("Time (ms)")
plt.ylabel("Firing rate (spikes/sec)")
plt.suptitle("Firing Rate Over Time\nw_self="+str(w_self)+", w_other="+str(w_other))
plt.legend()
plt.show()
#plt.plot(v1a_vect, v1b_vect)
#plt.show()


#Nullcline
#r1 = w12*r2 + I1
#r2 = w21*r1 + I2
'''
fig = plt.figure()
ax = plt.subplot(1,1,1)
def plotNullcline(w12):
    ax.clear()
    w21 = None
    if w12 == 0:
        return None
    elif w12 < 0:
        w21 = 1/w12
    else:
        w21 = 1/w12
    #r1 = np.linspace(0,70,140)
    #r2 = np.linspace(0,70,140)
    r1 = np.linspace(-70,70,140)
    r2 = np.linspace(-70,70,140)

    I1 = np.zeros(len(r1)) + 0
    I2 = np.zeros(len(r1)) + 0

    r1Null = w12*r2 + I1
    r2Null = w21*r1 + I2
    #ax.set_xlim(0,70)
    #ax.set_ylim(0,70)
    ax.plot(r1, r2Null, label='r2 Nullcline')
    rareR1Null = [r1Null[r] for r in range(0,len(r1Null),3)]
    rareR2 = [r2[r] for r in range(0,len(r2),3)]
    ax.scatter(rareR1Null, rareR2, color='red',label='r1 Nullcline')
    ax.grid()
    ax.set_xlabel("r1")
    ax.set_ylabel("r2")
    ax.set_title("Line Attractor Infinite Fixed Points\nw12: " + str(round(w12,2)) + " w21: " + str(round(w21,2)))
    ax.legend()
    plt.xlim(-70,70)
    plt.ylim(-70,70)

animation = FuncAnimation(fig,func=plotNullcline, frames=np.arange(-1,2,.1))
writergif = anim.PillowWriter(fps=30)
animation.save('LineAttractor.gif', writer='pillow') 
#plt.show()
'''
#Winner takes all
current1a_vect = ConstCurrent(t_vect, 63, [t_stimStart, t_stimEnd]) #Current injected into Neuron 1 [spikes/sec] 63
current1b_vect = ConstCurrent(t_vect, 57, [t_stimStart, t_stimEnd]) #Current injected into Neuron 2 [spikes/sec] 57
w_self = 1
w_other = -10
weightMatrix = np.array([[w_self,w_other],
                        [w_other,w_self]]) #Adjustable weight matrix (-1 to 1 preferrable)
v1a_vect = np.zeros(len(t_vect)) #Output of Neuron 1 [spikes/sec]
v1b_vect = np.zeros(len(t_vect)) #Output of Neuron 2 [spikes/sec]
v1a_vect[0] = 63
v1b_vect[0] = 57
tIdx = 1 #Time index used for iteration
r_max = 200
#r_max1 = 200
#r_max2 = 200
#maxxed1=False
#maxxed2=False
while tIdx < len(t_vect):
    #Calculating current in Neuron 1
    #u1a_vect = [0,v1b_vect[tIdx-1]] #Input of each neuron
    #totalInput1a = current1a_vect[tIdx-1] + sum(np.dot(u1a_vect, weightMatrix)) #Total input including decay and injected current
    #totalInput1a = sum(np.dot(u1a_vect, weightMatrix))
    #v1a_vect[tIdx] = v1a_vect[tIdx-1] + dt/tau*(-v1a_vect[tIdx-1] + totalInput1a) #Stepwise calculation of the next firing rate

    #Calculating current in Neuron 2
    #See previous comment
    #u1b_vect = [v1a_vect[tIdx-1],0] 
    #totalInput1b =  current1b_vect[tIdx-1] + sum(np.dot(u1b_vect, weightMatrix))
    #totalInput1b = sum(np.dot(u1b_vect, weightMatrix))
    #v1b_vect[tIdx] = v1b_vect[tIdx-1] + dt/tau*(-v1b_vect[tIdx-1] + totalInput1b)
    u = np.array([v1a_vect[tIdx-1],v1b_vect[tIdx-1]])
    i = np.array([current1a_vect[tIdx-1],current1b_vect[tIdx-1]])
    #if tIdx < 10:
    #    print("Input: " + str(u))
    v_vect_t = u + dt/tau*(-u + np.dot(u, weightMatrix) + i)
    '''
    if tIdx < 10:
        print("Dot: " + str(np.dot(u,weightMatrix)))
        print("Sum: " + str(-u + np.dot(u,weightMatrix)))
        print("Change: " + str(dt/tau*(-u + np.dot(u, weightMatrix))))
        print("Final: " + str(v_vect_t))
    if tIdx%100 == 0:
        print(v_vect_t)'''
    #v1a_vect[tIdx] = min(max(0,v_vect_t[0]), r_max)
    v1a_vect[tIdx] = max(0,v_vect_t[0])
    '''if v1a_vect[tIdx] > r_max1:
        maxxed1=True
        v1b_vect[tIdx] = min(r_max1, v1b_vect[tIdx])
    else:
        maxxed1=False'''
    
    #v1b_vect[tIdx] = min(max(0,v_vect_t[1]), r_max)
    v1b_vect[tIdx] = max(0,v_vect_t[1])
    '''if v1b_vect[tIdx] > r_max1:
        maxxed2=True
        v1b_vect[tIdx] = min(r_max2, v1b_vect[tIdx])
    else:
       maxxed2=False
    if maxxed1:
        r_max1 = r_max1 - 20
        if r_max1 < 0:
            maxxed1=False
            r_max1=200
        if tIdx%50 == 0:
            print(r_max1)
    if maxxed2:
        r_max2 = r_max2 - 20
        if r_max2 < 0:
            maxxed2=False
            r_max2=200'''
    
    #v1a_vect[tIdx] = v_vect_t[0]
    #v1b_vect[tIdx] = v_vect_t[1]
    #Increase the time index
    tIdx = tIdx + 1
plt.plot(t_vect, v1a_vect, label="Neuron 1 (63Hz)")
plt.plot(t_vect, v1b_vect, label="Neuron 2 (57Hz)")
plt.suptitle("Winner-Takes-All (Runaway Growth)\nw_self="+str(w_self)+", w_other="+str(w_other))
plt.xlabel("Time [ms]")
plt.ylabel("Firing Rate [spikes/sec]")
plt.legend()
plt.show()
