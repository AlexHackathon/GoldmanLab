import random
import numpy
import numpy as np
import matplotlib.pyplot as plt
import math
import Helpers.ActivationFunction as ActivationFunction
import Helpers.GraphHelpers as MyGraphing
import Helpers.CurrentGenerator as MyCurrent

class Simulation:
    def __init__(self, neuronNum, dt, stimStart, stimEnd, end, tau, a, p, maxFreq, eyeStartParam, eyeStopParam,
                 eyeResParam):
        '''Instantiates the simulation
           ---------------------------
           Parameters
           neuronNum: the number of neurons simulated
           dt: the time step of the simulation in ms
           t_stimStart: the start of the electric current stimulation
           t_stimEnd: the end of the electric current stimulation
           t_end: the end of the simulation
           tau: the neuron time constant for all neurons
           currentMatrix: a matrix with each row representing the injected current of a neuron over time
           a: the a parameter of the system's nonlinearity
           p: the p parameter of the system's nonlinearity'''
        self.neuronNum = neuronNum  # Number of neurons in the simulation
        self.dt = dt  # Time step [ms]
        self.t_stimStart = stimStart  # Stimulation start [ms]
        self.t_stimEnd = stimEnd  # Stimulation end [ms]
        self.t_end = end  # Simulation end [ms]
        self.tau = tau
        self.t_vect = np.arange(0, self.t_end, self.dt)

        self.eyeStart = eyeStartParam
        self.eyeStop = eyeStopParam
        self.eyeRes = eyeResParam

        self.current_mat = np.zeros((self.neuronNum, len(self.t_vect)))  # Defaults to no current
        # self.w_mat = np.zeros((self.neuronNum, self.neuronNum + 1)) #Tonic as last column
        self.w_mat = np.zeros((self.neuronNum, self.neuronNum))
        self.eta = None
        self.ksi = None
        self.v_mat = np.zeros((len(self.t_vect), neuronNum))  # For simulation

        self.maxFreq = maxFreq
        self.onPoints = np.linspace(self.eyeStart, self.eyeStop, self.neuronNum + 1)[:-1]
        self.eyePos = np.linspace(self.eyeStart, self.eyeStop, self.eyeRes)
        self.r_mat = None
        self.T = -(self.maxFreq / (self.eyeStop - self.eyeStart))*self.onPoints

        self.fixedA = a
        self.fixedP = p
        # Fixed points is a list of the same length as eyePos
        # Each index contains a list of fixed points for that simulation
        # Each fixed point contains the values for each of the neurons at that time point
        self.fixedPoints = []  # NOT A SQUARE MATRIX