import numpy as np
import matplotlib.pyplot as plt
from randmap import rand_cmap as rm

""" Parameters/containers """
T = 100                        # Total duration
dt = 0.1                        # Integration time step
time = np.arange(0, T, dt)      # Time array
V = np.zeros(len(time)+1)       # Membrane voltage                   
x = np.ones(len(time)+1)        # input
y = np.zeros(len(time)+1)       # Decoder
gam = 0.05                      # Readout weights
Thr = (gam**2)/2                # Threshold
spiked = 0                      # Spikes
decay = 0.5                     # Decay constant
print("Threshold:", Thr)

""" Voltage ODE """
def dV(i, spike):
    return - decay *  V[i] + gam*(x[i]+(x[i]-x[i-1])/dt) - gam*gam*spike/dt

""" Decoder ODE """
def dy(i, spike):
    return - decay * y[i] + gam*spike/dt

""" Simulation """
for i,t in enumerate(time):
    """ Euler integration """
    V[i] = V[i-1] + dt * dV(i-1, spiked)
    y[i] = y[i-1] + dt * dy(i-1, spiked)
    
    """ Check spikes """    
    spiked = (V[i] > Thr)

""" Plotting """
time = np.arange(0, T+dt, dt)
plt.plot(time, y, color='#cc1111', label='$\hat{x}$')
plt.plot(time, x, 'k-', label = '$x$')
plt.plot(time, V, color='#00cc66', label = '$V$')
plt.plot([0,T],[Thr,Thr], color='#eeeeee')
plt.ylim([-0.05, 1.05])
plt.legend()
plt.show()      