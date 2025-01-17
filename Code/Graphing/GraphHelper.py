# MOVE FUNC TO TUNING CURVES
def PlotTargetCurves(self):
    '''Plot given target curves over eye position.'''
    for r in range(len(self.r_mat)):
        plt.plot(self.eyePos, self.r_mat[r])
    plt.xlabel("Eye Position")
    plt.ylabel("Firing Rate")
    plt.show()
def GraphWeightMatrix(self):
    '''Create a heat map of the weight matrix (nxn).

    Rows are postsynaptic neurons. Columns are pre-synaptic neurons.'''
    plt.imshow(self.w_mat)
    plt.title("Weight Matrix")
    plt.colorbar()
def PlotFixedPointsOverEyePosRate(self,neuronArray):
    '''Plots synaptic activation decay and growth.

    Uses the prediction of firing rate vs actual firing rate
    to visualize fixed points of the network.'''
    for n in neuronArray:
        r = np.zeros(len(self.eyePos))
        for e in range(len(self.eyePos)):
            r[e] = np.dot(self.w_mat[n], self.f(self.r_mat[:,e])) + self.T[n]
            r[e] = max(0, r[e])
        #plt.plot(self.eyePos, self.r_mat[n], label = "decay")
        plt.plot(self.eyePos, r, label = "growth")
        plt.xlabel("Eye Position")
        plt.ylabel("Fixed Points")
def PlotContribution(self):
    #Draw a line of the eye position
    #Right contribution is the weight matrix of the first half times the s of the first half
    #Left contribution is the opposite
    #Plot eye predicted, eye actual, left, and right
    zeros = np.zeros(len(self.eyePos)//2)
    eyeHalf = self.eyePos[len(self.eyePos)//2:]
    goalR = []
    for z in zeros:
        goalR.append(z)
    for e in eyeHalf:
        goalR.append(e)
    goalR = np.array(goalR)
    eR = np.zeros((len(self.eyePos)))
    eL = np.zeros((len(self.eyePos)))
    for e in range(len(self.eyePos)):
        sR = self.f(self.r_mat[:self.neuronNum//2,e])
        sL = self.f(self.r_mat[self.neuronNum//2:, e])
        eR[e] = np.dot(self.predictW[:self.neuronNum//2], sR)
        eL[e] = np.dot(self.predictW[self.neuronNum//2:], sL)
    plt.plot(self.eyePos, self.eyePos, label="Reference")
    plt.plot(self.eyePos, [self.PredictEyePosNonlinearSaturation(self.f(self.r_mat[:,e])) for e in range(len(self.eyePos))], label="Prediction")
    plt.plot(self.eyePos, eR, label="Right")
    plt.plot(self.eyePos, eL, label="Left")
    plt.plot(self.eyePos, [self.predictT for e in range(len(self.eyePos))], label="Tonic")
    plt.legend()
    plt.show()