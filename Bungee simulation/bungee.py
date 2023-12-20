import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Definiera högerledet
def ode(t, u, m, g, k, L):
    u1 = u[1]
    cd = 0.225      # friktionskoefficient
    c = cd*u1**2    # luftmotstånd
    
    # luftmotståndet har motsatt riktning som hastigheten
    riktning = np.sign(u1)
    
    # Hookes lag gäller när repet sträckts ut
    if(u[0] <= L):
        u2 = (m*g - riktning*c)/m
    else:
        u2 = (m*g - riktning*c - k*(u[0]-L))/m
    return [u1, u2] # [hastighet, acceleration]

# 4-stegs Runge-Kutta
def rk4(f, tspan, y0, h, *args):   
    interval=round((tspan[1]-tspan[0])/h)
    tvec=np.linspace(tspan[0],tspan[1],interval+1)
    y=np.zeros((len(tvec),len(y0)))
    i=0
    y[i,:]=y0
    for t in tvec[0:len(tvec)-1]:
        k1=f(t,y[i,:], *args)
        k2=f(t + h/2, y[i,:] + np.multiply((h/2),k1), *args) 
        k3=f(t + h/2, y[i,:] + np.multiply((h/2),k2), *args)
        k4=f(t + h,   y[i,:] + np.multiply(h,k3),     *args)

        y[i+1,:]=y[i,:]+ np.multiply((h/6), (np.add(np.add(k1, k2), np.add(k3, k4))))
        i+=1
    return tvec, y

# Definiera parametrar
m = 150         # hopparens vikt
g = 9.82        # gravitation
k = 220         # fjäderkonstant
L = 20          # repets längd

t0 = 0.0
t1 = 200.0
tspan = (t0, t1)
u0 = [0, 0]
h = 0.1         # tidssteg

# Plotta lösningen
t, u = rk4(ode, tspan, u0, h, m, g, k, L)
plt.plot(t, -u[:, 0], '-r')
plt.title('Bungee Jumping')
plt.xlabel('t (sekunder)')
plt.ylabel('Position')
plt.show()