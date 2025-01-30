import numpy as np
def BoundQuadrants(n, wMin, wMax, neuronNum):
    '''Return a vector of restrictions based on same side excitation
    and opposite side inhibition.'''
    upperBounds = [0 for n in range(neuronNum + 1)]
    lowerBounds = [0 for n in range(neuronNum + 1)]
    if n < neuronNum // 2:
        for nIdx in range(neuronNum + 1):
            if nIdx < neuronNum // 2:
                upperBounds[nIdx] = wMax
                lowerBounds[nIdx] = 0
            else:
                upperBounds[nIdx] = 0
                lowerBounds[nIdx] = wMin
    else:
        for nIdx in range(neuronNum+1):
            if nIdx < neuronNum // 2:
                upperBounds[nIdx] = 0
                lowerBounds[nIdx] = wMin
            else:
                upperBounds[nIdx] = wMax
                lowerBounds[nIdx] = 0
    upperBounds[-1] = np.inf
    lowerBounds[-1] = -np.inf
    return (lowerBounds, upperBounds)