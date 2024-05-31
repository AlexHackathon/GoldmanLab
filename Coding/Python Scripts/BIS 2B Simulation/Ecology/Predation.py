import numpy as np
import matplotlib.pyplot as plt
dt = .001

r = 2
K = 60
a = .1
c = 2
m  = 2

refuge = 0

t_vect = np.arange(0,100,dt)
N = np.zeros(len(t_vect))
P = np.zeros(len(t_vect))
N[0] = 15
P[0] = 1
tIdx = 1

while tIdx < len(t_vect):
    '''print(r*N[tIdx-1]*(1-N[tIdx-1]/K))
    print(a*N[tIdx-1]*P[tIdx-1])
    print(c*a*N[tIdx-1]*P[tIdx-1])
    print(m*P[tIdx-1])'''
    #N[tIdx] = N[tIdx-1] + (r*N[tIdx-1]*(1-N[tIdx-1])/K)-a*N[tIdx-1]*P[tIdx-1])
    N[tIdx] = N[tIdx-1] + dt*((r*N[tIdx-1])-a*N[tIdx-1]*P[tIdx-1])
    P[tIdx] = P[tIdx-1] + dt*(c*a*N[tIdx-1]*P[tIdx-1]-m*P[tIdx-1])
    N[tIdx] = max(refuge,N[tIdx])
    tIdx = tIdx+1
plt.plot(t_vect, N, label="Prey")
plt.plot(t_vect, P, label="Predator")
plt.legend()
plt.show()

r=8
while tIdx < len(t_vect):
    '''print(r*N[tIdx-1]*(1-N[tIdx-1]/K))
    print(a*N[tIdx-1]*P[tIdx-1])
    print(c*a*N[tIdx-1]*P[tIdx-1])
    print(m*P[tIdx-1])'''
    #N[tIdx] = N[tIdx-1] + (r*N[tIdx-1]*(1-N[tIdx-1])/K)-a*N[tIdx-1]*P[tIdx-1])
    N[tIdx] = N[tIdx-1] + dt*((r*N[tIdx-1])-a*N[tIdx-1]*P[tIdx-1])
    P[tIdx] = P[tIdx-1] + dt*(c*a*N[tIdx-1]*P[tIdx-1]-m*P[tIdx-1])
    N[tIdx] = max(refuge,N[tIdx])
    tIdx = tIdx+1
plt.plot(t_vect, N, label="Prey")
plt.plot(t_vect, P, label="Predator")
plt.legend()
plt.show()
