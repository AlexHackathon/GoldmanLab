import numpy as np
def Sigmoid(input):
    return
def Geometric(input, a, p):
    x = np.array(input)
    numerator = x ** p
    denominator = a + x ** p
    return numerator / denominator
def RELU(input):
    return
def Stepwise(input, threshold):
    return
def SynapticSaturation(input, alpha):
    input = np.array(input)
    numerator = alpha * input
    denominator = 1 + alpha * input
    return numerator/denominator
def SynapticFacilitation(input, P_0, F_f, t_p):
    '''Return the activation function for synaptic facilitation'''
    numerator = P_0+F_f*t_p*input
    denominator = 1+F_f*t_p*input
    return  numerator/denominator * input
def SynapticFacilitationCa(input, P_0, F_f, t_p, n, rStar=200):
    '''Return the activation function for synaptic facilitation'''
    fixInput = np.divide(input, rStar)
    fixInput = fixInput ** n
    numerator = P_0 + F_f * t_p * fixInput
    denominator = 1 + F_f * t_p * fixInput
    return numerator / denominator * fixInput
def SynapticFacilitationNoR(input, P_0, F_f, t_p):
    '''Return the activation function for synaptic facilitation'''
    numerator = P_0 + F_f * t_p * input
    denominator = 1 + F_f * t_p * input
    return numerator / denominator
def SynapticDepressionNoR(input, P_0, F_d, t_p):
    '''Return the activation function for synaptic depression'''
    numerator = P_0
    denominator = 1 + (1-F_d) * input * t_p
    return numerator / denominator
def SynapticDepression(input, P_0, F_d, t_p):
    '''Return the activation function for synaptic depression'''
    numerator = P_0
    denominator = 1 + (1-F_d) * input * t_p
    return numerator / denominator * input