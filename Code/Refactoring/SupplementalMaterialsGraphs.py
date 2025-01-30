import matplotlib.pyplot as plt
from matplotlib.gridspec import  GridSpec
import numpy as np
import Refactoring.SimSupport as SimSupport

def SupplementalGraphsFacilitation(sim):
    #Set title
    plt.suptitle("Facilitation Simulation")
    plt.title("TauF: " + str(sim.t_f) + " f: " + str(sim.f_f) + " P0: " + str(sim.P0_f))
    #Graphing
    grid = GridSpec(6, 6)
    fig = plt.figure()
    #Synaptic Activation curves
    ax1=fig.add_subplot(grid[0,0])
    r = np.linspace(0,100,100)
    s = sim.f(r)
    plt.plot(r,s)

    ax2=fig.add_subplot(grid[0,1])
    r = np.linspace(0, 100, 100)
    s = sim.f(r)
    plt.plot(r, s)

    #Presynaptic Neuron Tuning Curves
    #First half
    ax3=fig.add_subplot(grid[1,0])
    for n in sim.r_mat[:sim.neuronNum//2,:]:
        plt.plot(sim.eyePos, n)

    #Second half
    ax4=fig.add_subplot(grid[1,1])
    for n in sim.r_mat[sim.neuronNum//2:,:]:
        plt.plot(sim.eyePos, n)

    #Presynaptic neuron activations
    #First half
    ax5=fig.add_subplot(grid[2,0])
    for n in sim.r_mat[:sim.neuronNum//2,:]:
        plt.plot(sim.eyePos, sim.f(n))

    #Second half
    ax6=fig.add_subplot(grid[2,1])
    for n in sim.r_mat[sim.neuronNum//2:,:]:
        plt.plot(sim.eyePos, sim.f(n))

    #For Single Neuron
    targetNeuron = 5
    #Individual Inputs With Weights
    #Excitatory
    ax7=fig.add_subplot(grid[3,0])
    rContributionE = np.zeros((sim.eyeRes,sim.neuronNum//2))
    s = sim.f(sim.r_mat[:sim.neuronNum//2,:])
    for i in range(len(s)):
        for j in range(len(s[i])):
            s[i,j] = s[i,j] * sim.w_mat[targetNeuron,i]
    for s_n in s:
        plt.plot(sim.eyePos,s_n)

    #Inhibitory
    ax8=fig.add_subplot(grid[3,1])
    rContributionI = np.zeros((sim.eyeRes,sim.neuronNum//2))
    s = sim.f(sim.r_mat[sim.neuronNum//2:,:])
    for i in range(len(s)):
        for j in range(len(s[i])):
            s[i,j] = s[i,j] * sim.w_mat[targetNeuron,i]
    for s_n in s:
        plt.plot(sim.eyePos,s_n)

    #Total Inputs
    #Excitatory
    ax9=fig.add_subplot(grid[4,0])
    s = sim.f(sim.r_mat[:sim.neuronNum//2,:])
    totalInputA = np.zeros((sim.eyeRes,))
    for e in range(sim.eyeRes):
        totalInputA[e] = np.dot(s[:,e],sim.w_mat[targetNeuron,:sim.neuronNum//2])
    plt.plot(sim.eyePos,totalInputA)

    #Inhibitory
    ax10=fig.add_subplot(grid[4,1])
    s = sim.f(sim.r_mat[sim.neuronNum//2:,:])
    totalInputB = np.zeros((sim.eyeRes,))
    for e in range(sim.eyeRes):
        totalInputB[e] = np.dot(s[:,e],sim.w_mat[targetNeuron,sim.neuronNum//2:])
    plt.plot(sim.eyePos,totalInputB)

    #Total All Inputs
    ax11=fig.add_subplot(grid[5,:1])
    total=totalInputA+totalInputB
    plt.plot(sim.eyePos, total)
    plt.plot(sim.eyePos, sim.r_mat[targetNeuron])
    plt.plot(sim.eyePos, sim.T[targetNeuron]*np.ones((len(sim.eyePos,))))
    plt.plot(sim.eyePos, total + sim.T[targetNeuron]*np.ones((len(sim.eyePos,))))

    ax12=fig.add_subplot(grid[:3,2:])
    #Run the simulation within the simulation class (Choose and store)
    timeAtKill = 200
    for e in range(len(sim.eyePos)):
        if e % 1000 == 0:
            print(e)
            mySimRes = sim.RunSimF(timeAtKill, startIdx=e)
            plt.plot(sim.t_vect, mySimRes[0], color="green")
            myDead = SimSupport.GetDeadNeurons(1, True, sim.neuronNum)
            mySimRes2 = sim.RunSimF(timeAtKill, startIdx=e, dead=myDead)
            plt.plot(sim.t_vect, mySimRes2[0], color="red")
    ax13=fig.add_subplot(grid[3:,2:])
    plt.imshow(sim.w_mat)
    plt.show()