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