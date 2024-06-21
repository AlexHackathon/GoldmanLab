import numpy as np
#Creates a box current over the time vector with magnitude stimMag
def ConstCurrent(time_vect, stimMag, stimStartEnd_vect):
    addCurrent = False
    i = 0
    current_vect = np.zeros(len(time_vect))
    for x in range(0, len(current_vect)): #len had -1 don't know why
        if i >= len(stimStartEnd_vect):
            continue
        elif time_vect[x] >= stimStartEnd_vect[i]:
            addCurrent = not addCurrent
            i = i + 1
        if addCurrent:
            current_vect[x] = stimMag
    return current_vect
def ConstCurrentMatEig(time_vect, eigenDataParam, stimStartEnd_vect):
    finalCurr = []
    for s in eigenDataParam.GetVect():
        finalCurr.append(ConstCurrent(time_vect, s * 6, stimStartEnd_vect))
    return np.array(finalCurr)
def ConstCurrentBursts(time_vect, mag, dur, isi, start, end, neurons):
    '''Creates a current matrix that shows the amount of input over time
    Rows are a neuron and its variation in input over time
    Columns are a time point and the input to all neurons at that point'''
    finalCurr = []
    isiElapsed = 0
    durElapsed = 0
    for n in range(neurons):
        curr = np.zeros(len(time_vect))
        durElapsed = 0
        isiElapsed = 0
        for i in range(len(time_vect)):
            if i < start or i > end:
                continue
            if durElapsed < dur:
                curr[i] = mag
            if durElapsed < dur:
                durElapsed += 1
            elif isiElapsed < isi:
                isiElapsed += 1
            elif isiElapsed >= isi and durElapsed >= dur:
                isiElapsed = 0
                durElapsed = 0
            else:
                print("Shouldn't be possible so there is a logic error with current.")
        finalCurr.append(curr)
    print(np.shape(curr))
    return np.array(finalCurr)
