import matplotlib.pyplot as plt
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
def ConstCurrentBursts(time_vect, mag, dur, isi, start, end, neurons, dt):
   '''Creates a current matrix that shows the amount of input over time
   Rows are a neuron and its variation in input over time
   Columns are a time point and the input to all neurons at that point'''
   finalCurr = []
   isiFramesElapsed = 0
   durFramesElapsed = 0
   for n in range(neurons):
       curr = np.zeros(len(time_vect))
       durFramesElapsed = 0
       isiFramesElapsed = 0
       for i in range(len(time_vect)):
           if i < start/dt or i > end/dt:
               continue
           if durFramesElapsed < dur / dt:
               curr[i] = mag
           if durFramesElapsed < dur / dt:
               durFramesElapsed += 1
           elif isiFramesElapsed < isi / dt:
               isiFramesElapsed += 1
           elif isiFramesElapsed >= isi/dt and durFramesElapsed >= dur/dt:
               isiFramesElapsed = 0
               durFramesElapsed = 0
           else:
               print("Shouldn't be possible so there is a logic error with current.")
       finalCurr.append(curr)
   print(np.shape(curr))
   return np.array(finalCurr)
def ShouldGetBurst(time_vect, dur, isi, start, end, dt):
   finalCurr = []
   isiFramesElapsed = 0
   durFramesElapsed = 0
   curr = np.zeros(len(time_vect))
   durFramesElapsed = 0
   isiFramesElapsed = 0
   for i in range(len(time_vect)):
       if i < start / dt or i > end / dt:
           continue
       if durFramesElapsed < dur / dt:
           curr[i] = True
       else:
           curr[i] = False
       if durFramesElapsed < dur / dt:
           durFramesElapsed += 1
       elif isiFramesElapsed < isi / dt:
           isiFramesElapsed += 1
       elif isiFramesElapsed >= isi / dt and durFramesElapsed >= dur / dt:
           isiFramesElapsed = 0
           durFramesElapsed = 0
       else:
           print("Shouldn't be possible so there is a logic error with current.")
   return np.array(curr)
def LiveInput(r_vect, c):
   return c * r_vect / np.linalg.norm(r_vect)
def PlotCurrent():
   t_vect = np.arange(0,1000,.1)
   x = ConstCurrentBursts(t_vect, .0005, 10, 100, 200, 500, 1,.1) #10 and 100 because dt=0.1ms
   plt.plot(t_vect, x[0])
   plt.show()