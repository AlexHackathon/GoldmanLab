def SetCurrentSplit(self, currentVect):
    '''Sets the current currentVect for first n/2 neurons and -currentVect for last n/2 neurons'''
    for n in range(self.neuronNum):
        if n < self.neuronNum // 2:
            self.current_mat[n] = currentVect
        else:
            self.current_mat[n] = -currentVect


def SetCurrentDoubleSplit(self, currentVect):
    '''Sets the current currentVect for first n/2 neurons and -currentVect for last n/2 neurons'''
    for n in range(self.neuronNum):
        if n < self.neuronNum // 2:
            self.current_mat[n] = currentVect
        else:
            self.current_mat[n] = -currentVect
    self.current_mat[:, len(self.t_vect) // 2:] = -self.current_mat[:, len(self.t_vect) // 2:]

def ReadWeightMatrix(self, wfl, ewfl,Tfl):
    readMatrix=pickle.load(open(wfl,"rb"))
    self.w_mat = readMatrix
    readMatrixT = pickle.load(open(Tfl,"rb"))
    self.T = readMatrixT
    readMatrixEye = pickle.load(open(ewfl,"rb"))
    self.predictW = readMatrixEye[:-1]
    self.predictT = readMatrixEye[-1]
def SynapticActivationCurves(self):
    r = np.linspace(0, self.maxFreq, 100)
    s = self.f(r)
    return r, s
def GetDeadNeurons(fraction, firstHalf, neuronNum):
    if firstHalf:
        return range(0, int(neuronNum // 2 * fraction))
    else:
        return [neuronNum // 2 + j for j in range(0, int(neuronNum // 2 * fraction))]
def FitHelperParallel(startN, endN, X, bounds, r_mat_neg, sim):
    for myN in range(startN,endN):
        r = r_mat_neg[myN]
        solution = None
        if bounds != None:
            bounds = bounds[myN]
            solution = scipy.optimize.lsq_linear(X, r, bounds)
        else:
            solution = scipy.optimize.lsq_linear(X, r)
        sim.w_mat[myN] = solution.x[:-1]
        sim.T[myN] = solution.x[-1]
def FitWeightMatrixExcludeParallel(fileLoc, eyeFileLoc, tFileLoc, sim, bounds):
    '''Fit fixed points in the network using target curves.

    Create an activation function matrix X (exn+1).
    Fit each row of the weight matrix with linear regression.
    Call the function to fit the predictor of eye position.
    Exclude eye positions where a neuron is at 0 for training each row.'''
    #POTENTIAL CHANGE: Take ten rows and execute them on a core
    #Update the program to say that those have been taken
    #When the other core finishes, it updates which it has taken
    #If the last has been taken, continue to writing
    X = np.ones((len(sim.eyePos), len(sim.r_mat) + 1))
    for i in range(len(X)):
        for j in range(len(X[0]) - 1):
            X[i, j] = sim.f(sim.r_mat[j, i])
    p1 = multiprocessing.Process(target=FitHelperParallel, args=(0, sim.neuronNum//2, X, bounds, sim.r_mat_neg, sim))
    p2 = multiprocessing.Process(target=FitHelperParallel, args=(sim.neuronNum//2, sim.neuronNum, X, bounds, sim.r_mat_neg, sim))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    sim.WriteWeightMatrix(sim.w_mat, fileLoc)
    sim.WriteWeightMatrix(sim.T, tFileLoc)
    sim.FitPredictorNonlinearSaturation(eyeFileLoc)