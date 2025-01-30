import scipy
import numpy as np
import multiprocessing as mp
def GetDeadNeurons(fraction, firstHalf, neuronNum):
    if firstHalf:
        return range(0, int(neuronNum // 2 * fraction))
    else:
        return [neuronNum // 2 + j for j in range(0, int(neuronNum // 2 * fraction))]
def FitHelperParallel(n, X, bounds, r_mat_neg):
    print(n)
    r = r_mat_neg[n]
    solution = None
    if bounds != None:
        solution = scipy.optimize.lsq_linear(X, r, bounds[n])
    else:
        solution = scipy.optimize.lsq_linear(X, r)
    return solution
def FitWeightMatrixExcludeParallel(sim, bounds):
    '''Fit fixed points in the network using target curves.

    Create an activation function matrix X (exn+1).
    Fit each row of the weight matrix with linear regression.
    Call the function to fit the predictor of eye position.
    Exclude eye positions where a neuron is at 0 for training each row.'''
    #POTENTIAL CHANGE: Take ten rows and execute them on a core
    #Update the program to say that those have been taken
    #When the other core finishes, it updates which it has taken
    #If the last has been taken, continue to writing
    print("Started fitting")
    X = np.ones((len(sim.eyePos), len(sim.r_mat) + 1))
    for i in range(len(X)):
        for j in range(len(X[0]) - 1):
            X[i, j] = sim.f(sim.r_mat[j, i])
    print("Finished X")
    """p1 = multiprocessing.Process(target=FitHelperParallel, args=(0, sim.neuronNum//2, X, bounds, sim.r_mat_neg, sim))
    p2 = multiprocessing.Process(target=FitHelperParallel, args=(sim.neuronNum//2, sim.neuronNum, X, bounds, sim.r_mat_neg, sim))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    sim.FitPredictorNonlinearSaturation()"""
    print("Started")
    pool = mp.Pool(5) #
    result = pool.starmap(FitHelperParallel, [(n,X,bounds, sim.r_mat_neg) for n in range(sim.neuronNum)])
    return  result
def FitWeightMatrixExclude(tuningCurves, tuningCurvesNeg, activationFunc, bounds):
    '''Fit fixed points in the network using target curves.

    Create an activation function matrix X (exn+1).
    Fit each row of the weight matrix with linear regression.
    Call the function to fit the predictor of eye position.
    Exclude eye positions where a neuron is at 0 for training each row.'''
    #POTENTIAL CHANGE: Take ten rows and execute them on a core
    #Update the program to say that those have been taken
    #When the other core finishes, it updates which it has taken
    #If the last has been taken, continue to writing
    X = np.ones((len(tuningCurves[1]), len(tuningCurves) + 1))
    for i in range(len(X)):
        for j in range(len(X[0]) - 1):
            X[i, j] = activationFunc(tuningCurves[j, i])
    w_mat = np.zeros((len(tuningCurves), len(tuningCurves)))
    t_vect = np.zeros(len(tuningCurves))
    for n in range(len(tuningCurves)):
        if(n%10==0):
            print(n)
        r = tuningCurvesNeg[n]
        solution = None
        if bounds != None:
            solution = scipy.optimize.lsq_linear(X, r, bounds[n])
        else:
            solution = scipy.optimize.lsq_linear(X, r)
        w_mat[n] = solution.x[:-1]
        t_vect[n] = solution.x[-1]
    return w_mat, t_vect

def GetDeadNeurons(fraction, firstHalf, neuronNum):
    if firstHalf:
        return range(0, int(neuronNum // 2 * fraction))
    else:
        return [neuronNum // 2 + j for j in range(0, int(neuronNum // 2 * fraction))]