import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def ode_rhs(t, y):
    yt = np.e**(t * np.sin(y))
    return yt

def trapets(f, tspan, u0, dt):
    interval = round((tspan[1] - tspan[0]) / dt)
    tvec = np.linspace(tspan[0], tspan[1], interval + 1)
    u = np.zeros((len(tvec), len(u0)))
    u[0, :] = u0

    for i in range(len(tvec) - 1):
        t = tvec[i]
        u[i + 1, :] = u[i, :] + (dt / 2) * (f(t, u[i, :]) + f(t + 1, u[i + 1, :]))

    return tvec, u

# Define initial conditions and parameters
t0 = 0.0          # Initial time
t1 = 3.0         # Final time
tspan = (t0, t1)  # Time interval
dt = 0.01         # Time step
v0 = [0]          # Initial

# Apply trapets method to solve the ODE
t, u = trapets(ode_rhs, tspan, v0, dt)
# y = fsolve((t, u), v0)
# plt.plot(y)

# Plot the solution
plt.plot(t, u[:, 0], '-b')
plt.title('Q4: Trapets')
plt.xlabel('t')
plt.show()