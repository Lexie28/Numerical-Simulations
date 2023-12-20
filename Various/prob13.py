from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt


# right hand side of the ODE
def ode_rhs(t, y) :
    yt = np.sin(t) - y
    return yt

def heuns(f, tspan, u0, dt):   
    interval=round((tspan[1]-tspan[0])/dt)
    tvec=np.linspace(tspan[0],tspan[1],interval+1)
    y=np.zeros((len(tvec),len(u0)))
    i=0
    y[i,:]= u0
    for t in tvec[0:len(tvec)-1]:

        k1=f(t,y[i,:])
        k2=f(t+dt,y[i,:]+dt*k1)
        y[i+1,:] = y[i,:]+(dt/2)*(k1+k2)
        i+=1
        
    return tvec, y

t0 = 0.0          # Starttid
t1 = 10.0         # sluttid
tspan = (t0, t1)  # Tidsintervall
dt = 0.01        # Tidsteg
u0 = [0]  
m = 85.0
cd = 0.225
t,u = heuns(ode_rhs, tspan, u0, dt)

plt.plot(t,u[:,0],'-b')
plt.show()