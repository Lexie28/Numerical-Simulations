import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation for the second-order system
def ode_rhs(t, S):
    x, v = S
    return [v, -v**2 + np.sin(x)]

# Define the Runge-Kutta (RK4) method function
def runge_kutta(f, tspan, u0, dt):
    # Calculate the number of intervals
    interval = round((tspan[1] - tspan[0]) / dt)
    # Create a time vector
    tvec = np.linspace(tspan[0], tspan[1], interval + 1)
    # Initialize the solution array
    u = np.zeros((len(tvec), len(u0)))
    u[0, :] = u0  # Set initial conditions

    i = 0

    # Perform the RK4 method iteration
    for t in tvec[0:len(tvec) - 1]:
        k1 = np.array(f(t, u[i, :]))
        k2 = np.array(f(t + 0.5 * dt, u[i, :] + 0.5 * dt * k1))
        k3 = np.array(f(t + 0.5 * dt, u[i, :] + 0.5 * dt * k2))
        k4 = np.array(f(t + dt, u[i, :] + dt * k3))
        u[i + 1, :] = u[i, :] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)  # Update the solution using RK4
        i += 1

    return tvec, u

# Define initial conditions and parameters
t0 = 0.0          # Initial time
t1 = 1.0         # Final time
tspan = (t0, t1)  # Time interval
dt = 0.01         # Time step
S0 = [0, 5]       # Initial state vector [x0, v0]

# Apply RK4 method to solve the ODE
t, u = runge_kutta(ode_rhs, tspan, S0, dt)

# Extract the solution components
x_sol = u[:, 0]
v_sol = u[:, 1]

# Plot the solution
plt.plot(t, x_sol, '-b', label='x(t)')
plt.plot(t, v_sol, '-r', label='v(t)')
plt.title("Coupled 2nd Order Differential Equations (RK4 Method)")
plt.xlabel('t')
plt.legend()
plt.grid(True)
plt.show()
