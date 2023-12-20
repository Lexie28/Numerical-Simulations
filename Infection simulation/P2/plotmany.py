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
V0 = 0

# smittspridningen
beta = 0.3

# återhämtningen
gamma = 1/7
alpha = 1/1 # nämnaren är antalet dagar
my = 1/100 # dödligheten
vacc = 1 # hur många som vaccineras per dag

def stoch ():
    sMat = np.array([ 
    [-1,  1,  0, 0, 0, 0],  # en blir exponerad
    [0,  -1,  1, 0, 0, 0],  # en bir sjuk
    [0, 0, -1, 1, 0, 0],   # en blir frisk
    [0, 0, -1, 0, 1, 0],   # en dör
    [-1, 0, 0, 0, 0, 1]
    ])
    return  sMat

def prop(X, coeff):
    S,E, I, R, D , V= X
    if S == 0: # om det inte längre finns någon att vaccinera
        return [
        (beta * S * I) / N  ,   
        alpha * E,
        gamma * I,     
        my*I,  
        0
        ]
    else:
        return[
        (beta * S * I) / N  ,   
        alpha * E,
        gamma * I,     
        my*I,  
        vacc   
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

    

X0 = [S0, E0, I0, R0, D0, V0]
coeff = (beta, gamma)
tspan = [0, 120]


num_iterations = 20  #antalet iterationer över algoritmen

plt.figure(figsize=(10, 6))


for _ in range(num_iterations):
    tvec, Xarr = SSA(prop, stoch, X0, tspan, coeff)
    S, E, I, R, D, V = Xarr.T
    
    # samma variableer har samma färg
    plt.plot(tvec, S, color='b', alpha=0.5)
    plt.plot(tvec, E,  color='g', alpha=0.5)
    plt.plot(tvec, I,  color='r', alpha=0.5)
    plt.plot(tvec, R,  color='c', alpha=0.5)
    plt.plot(tvec, D, color='m', alpha=0.5)
    plt.plot(tvec, V,  color='y', alpha=0.5)

plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.title(f'Stochastic Simulation of Disease Spread ({num_iterations} Iterations)')
plt.grid(True)
plt.show()
