import pickle
import matplotlib.pyplot as plt
import numpy as np

A, B, t1, t2= pickle.load(open("FitParams.bin", "rb"))
A = A[0:4,0:4]
B = B[0:4,0:4]
t1 = t1[0:4,0:4]
t2 = t2[0:4,0:4]
print(A)
print(B)
print(t1)
print(t2)

e = [-20.0, -9.997499374843711, 0.00500125031257781, 10.007501875468865]
functionFrac = [.8,.5,.2,.01]

fig, ax = plt.subplots(2,2)

for i in range(len(A)):
    ax[0,0].plot(e,A[i], label=str(functionFrac[i]) + " intact")
for j in range(len(A)):
    ax[0,1].plot(e,B[j], label=str(functionFrac[j]) + " intact")
for k in range(len(A)):
    ax[1,0].plot(e,t1[k], label=str(functionFrac[k]) + " intact")
for l in range(len(A)):
    ax[1,1].plot(e,t2[l], label=str(functionFrac[l]) + " intact")

ax[0,0].legend()
ax[0,0].set_title("Synaptic Tau Exponential Constant A vs Eye Position")
ax[0,0].set_xlabel("Eye Position")
ax[0,0].set_ylabel("A")

ax[0,1].legend()
ax[0,1].set_title("Facilitation Tau Exponential Constant B vs Eye Position")
ax[0,1].set_xlabel("Eye Position")
ax[0,1].set_ylabel("B")

ax[1,0].legend()
ax[1,0].set_title("Synaptic Tau vs Eye Position")
ax[1,0].set_xlabel("Eye Position")
ax[1,0].set_ylabel("Synaptic Tau")

ax[1,1].legend()
ax[1,1].set_title("Facilitation Tau vs Eye Position")
ax[1,1].set_xlabel("Eye Position")
ax[1,1].set_ylabel("A")

plt.tight_layout()
plt.show()