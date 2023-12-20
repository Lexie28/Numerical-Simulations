import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation for the second-order system
def ode_rhs(x, S):
    x, v = S
    return [v, -v**2 + np.sin(x)]

# Define the Euler method function
def euler(f, tspan, u0, dt):
    # Calculate the number of intervals
    interval = round((tspan[1] - tspan[0]) / dt)
    # Create a time vector
    tvec = np.linspace(tspan[0], tspan[1], interval + 1)
    # Initialize the solution array
    u = np.zeros((len(tvec), len(u0)))
    u[0, :] = u0  # Set initial conditions

    i = 0

    # Perform the Euler method iteration
    for t in tvec[0:len(tvec) - 1]:
        k1 = np.array(f(t, u[i, :]))  # Calculate the slope at the current point
        u[i + 1, :] = u[i, :] + dt * k1  # Update the solution using Euler's method
        i += 1

    return tvec, u

# Define initial conditions and parameters
t0 = 0.0          # Initial time
t1 = 1.0         # Final time
tspan = (t0, t1)  # Time interval
dt = 0.01         # Time step
S0 = [0, 5]       # Initial state vector [x0, v0]

# Apply Euler's method to solve the ODE
t, u = euler(ode_rhs, tspan, S0, dt)

# Extract the solution components
x_sol = u[:, 0]
v_sol = u[:, 1]

# Plot the solution
plt.plot(t, x_sol, '-b', label='x(t)')
plt.plot(t, v_sol, '-r', label='v(t)')
plt.title('Coupled 2nd Order Differential Equations (Euler\'s Method)')
plt.xlabel('t')
plt.legend()
plt.grid(True)
plt.show()
