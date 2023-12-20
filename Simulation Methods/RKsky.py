import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation for the skydiver's velocity
def ode_rhs(t, v, m, cd):
    g = 9.81  # Acceleration due to gravity
    c = cd * v  # Drag force
    vt = g - (c / m) * v  # Equation of motion
    return vt

# Define the RK4 method function
def rk4(f, tspan, u0, dt, *args):
    # Calculate the number of intervals
    interval = round((tspan[1] - tspan[0]) / dt)
    # Create a time vector
    tvec = np.linspace(tspan[0], tspan[1], interval + 1)
    # Initialize the solution array
    u = np.zeros((len(tvec), len(u0)))
    u[0, :] = u0  # Set initial conditions

    # Perform the RK4 method iteration
    for i in range(len(tvec) - 1):
        t = tvec[i]
        k1 = f(t, u[i, :], *args)  # Calculate the slope at the current point
        k2 = f(t + 0.5 * dt, u[i, :] + 0.5 * dt * k1, *args)  # Calculate the slope at mid-point 1
        k3 = f(t + 0.5 * dt, u[i, :] + 0.5 * dt * k2, *args)  # Calculate the slope at mid-point 2
        k4 = f(t + dt, u[i, :] + dt * k3, *args)  # Calculate the slope at the next point
        u[i + 1, :] = u[i, :] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # Update the solution

    return tvec, u

# Define initial conditions and parameters
t0 = 0.0          # Initial time
t1 = 15.0         # Final time
tspan = (t0, t1)  # Time interval
dt = 0.01         # Time step
v0 = [0]          # Initial velocity
m = 85.0          # Mass of the skydiver
cd = 0.225        # Drag coefficient

# Apply RK4 method to solve the ODE
t, u = rk4(ode_rhs, tspan, v0, dt, m, cd)

# Plot the solution
plt.plot(t, u[:, 0], '-b')
plt.show()
