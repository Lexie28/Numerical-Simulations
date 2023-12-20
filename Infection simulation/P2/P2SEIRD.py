import numpy as np
import random
import math
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#==================================
# Gillespies algoritm
# Numeriska metoder och Simulering
#==================================
# Populationen
N = 1000

# begynnelsevärden
S0 = 995
E0 = 5
I0 = 0
R0 = 0
D0 = 0

# smittspridningen
beta = 0.3

# återhämtningen
gamma = 1/7
alpha = 1/2 # nämnaren är antalet dagar
my = 1/40  #dödligheten

def stoch ():
    sMat = np.array([ 
    [-1,  1,  0, 0, 0],  # en blir exponerad
    [0,  -1,  1, 0, 0],  # en bir sjuk
    [0, 0, -1, 1, 0],   # en blir frisk
    [0, 0, -1, 0, 1],   # en dör
    ])
    return  sMat

def prop(X, coeff):
    S,E, I, R, D = X
    return [
        (beta * S * I) / N ,   
        alpha * E,
        gamma * I,     
        my*I      
    ]


def SSA(prop,stoch,X0,tspan,coeff):
    tvec=np.zeros(1)
    tvec[0]=tspan[0]
    Xarr=np.zeros([1,len(X0)])
    Xarr[0,:]=X0
    t=tvec[0]
    X=X0
    sMat=stoch()
    while t<tspan[1]:
        re=prop(X,coeff)
        a0=sum(re)
        if a0>1e-10:
            r1=random.random()
            r2=random.random()
            tau=-math.log(r1)/a0
            cre=np.cumsum(re)
            cre=cre/cre[-1]
            r=0
            while cre[r]<r2:
                r+=1
            t+=tau
            tvec=np.append(tvec,t)
            X=X+sMat[r,:]
            Xarr=np.vstack([Xarr,X])
        else:
            print('Simulation ended at t=', t)
            return tvec, Xarr
            
            
    return tvec, Xarr

    

X0 = [S0, E0, I0, R0, D0]
coeff = (beta, gamma)
tspan = [0, 120]


tvec, Xarr = SSA(prop, stoch, X0, tspan, coeff)

S, E, I, R, D = Xarr.T

plt.figure(figsize=(10, 6))
plt.plot(tvec, S, label='Susceptible')
plt.plot(tvec, E, label='exposed')
plt.plot(tvec, I, label='Infectious')
plt.plot(tvec, R, label='Recovered')
plt.plot(tvec, D, label='DEAD')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.title('Stochastic Simulation of Disease Spread (Gillespie Algorithm)')
plt.grid(True)
plt.show()
