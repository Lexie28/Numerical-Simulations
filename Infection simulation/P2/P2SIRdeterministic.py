import numpy as np
import random
import math
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Populationen
N = 1000

# begynnelsevärden
S0 = 995 #susceptible
I0 =5 #infected
R0 = 0 #recovered

# smittspridningen
beta = 0.3

# återhämtningen
gamma = 1/7

# spannet (0 til 120 dagar)
t = np.linspace(0, 120, 121)  # inkluderar 0 och 120


def model(y, t, beta, gamma, N):
    N = 1000
    S, I, R = y
    dSdt = -beta * I * S / N
    dIdt = (beta * I * S / N) - (gamma * I)
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Initial conditions as a list
y0 = [S0, I0, R0]

# Solve the ODEs
result = odeint(model, y0, t, args=(beta, gamma, N))

# Extract S, I, and R values from the result
S, I, R = result.T

plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infectious')
plt.plot(t, R, label='Recovered')
plt.xlabel('Tid (dagar)')
plt.ylabel('Population')
plt.legend()
plt.title('Simulation av smittspridning')
plt.grid(True)
plt.show()

import numpy as np
import random
import math
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt