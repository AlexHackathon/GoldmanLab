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
