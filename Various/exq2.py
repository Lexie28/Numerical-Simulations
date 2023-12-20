import numpy as np
import matplotlib.pyplot as plt


def ode_rhs(t,y):
    C = 1.8
    B= 10000
    yt = C*y*(1-(y/B))
    return yt

def rk4(f, tspan, u0, dt):
    interval = round((tspan[1] - tspan[0]) / dt)
    tvec = np.linspace(tspan[0], tspan[1], interval + 1)
    u = np.zeros((len(tvec), len(u0)))
    u[0, :] = u0

    for i in range(len(tvec) - 1):
        t = tvec[i]
        k1 = f(t, u[i, :])
        k2 = f(t + 0.5 * dt, u[i, :] + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, u[i, :] + 0.5 * dt * k2)
        k4 = f(t + dt, u[i, :] + dt * k3)
        u[i + 1, :] = u[i, :] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return tvec, u

t0 = 0.0          # Initial time
t1 = 8.0         # Final time
tspan = (t0, t1)  # Time interval
dt = 0.1         # Time step
y0 = [10]          # Initial value

t, u = rk4(ode_rhs, tspan, y0, dt)

# Plot the solution
plt.plot(t, u[:, 0], '-b')
plt.title('Q2: Runge-Kutta Bakterietillväxt')
plt.xlabel('t (sekunder)')
plt.ylabel('Bakterietillväxt')
plt.show()
