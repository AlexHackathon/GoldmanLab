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

print(SynapticFacilitation(np.array([1,2,3,4,5]), .1, .4, 50))