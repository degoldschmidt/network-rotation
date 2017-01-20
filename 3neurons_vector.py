import numpy as np
import matplotlib.pyplot as plt
from randmap import rand_cmap as rm

""" Parameters/containers """
N    = 3                            # Number of neurons
T    = 100                          # Total duration
dt   = 0.1                          # Integration time step
time = np.arange(0, T, dt)          # Time array
V    = np.zeros((N, len(time)+1))   # Membrane voltage                   
x    = np.ones(len(time)+1)         # Input
y    = np.zeros(len(time)+1)        # Decoder
gam  = 0.5*np.ones(N)/N             # Readout weights
mu   = 0.001                        # Regularization constant
Thr  = np.dot(gam,gam)/2 + mu/2     # Threshold
decay = 1                           # Decay constant
noise = np.zeros(N)  # Noise term
var = 0.001                         # Noise variance
print("Threshold:", Thr)        

""" Voltage ODE """
def dV(i, spike):
    return - decay * V[:,i] + gam*(x[i]+(x[i]-x[i-1])/dt) - (gam.T*gam + mu*np.eye(N)).dot(spike)/dt + noise

""" Decoder ODE """
def dy(i, spike):
    return -decay * y[i] + gam.dot(spike)/dt

""" Simulation """
for i,t in enumerate(time):
    """ Check spikes """    
    spiked = (V[:,i] > Thr)
        
    """ Euler integration """
    noise = var * np.random.randn(N) 
    V[:,i+1] = V[:,i] + dt * dV(i, spiked)
    y[i+1] = y[i] + dt * dy(i, spiked)

""" Plotting """
rmap = rm(100, type='bright', first_color_black=False, last_color_black=False, verbose=False)
time = np.arange(0, T+dt, dt)
plt.plot(time, x, 'k-', label = '$x$')
plt.plot(time, y, color='#bbbbbb', label='$\hat{x}$')
for neuron in range(N):
    name = '$V_{'+str(neuron+1)+'}$'
    plt.plot(time, V[neuron,:], color=rmap(neuron), label = name)
plt.legend()
plt.show()      