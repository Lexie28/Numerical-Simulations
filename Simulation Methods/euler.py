import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation for the skydiver's velocity
def ode_rhs(t, v, m, cd):
    g = 9.81  # Acceleration due to gravity
    c = cd * v  # Drag force
    vt = g - (c / m) * v  # Equation of motion
    return vt

# Define the Euler method function
def euler(f, tspan, u0, dt, *args):   
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
        k1 = f(t, u[i, :], *args)  # Calculate the slope at the current point
        u[i + 1, :] = u[i, :] + dt * k1  # Update the solution using Euler's method
        i += 1

    return tvec, u

# Define initial conditions and parameters
t0 = 0.0          # Initial time
t1 = 15.0         # Final time
tspan = (t0, t1)  # Time interval
dt = 0.01         # Time step
v0 = [0]          # Initial velocity
m = 85.0          # Mass of the skydiver
cd = 0.225        # Drag coefficient

# Apply Euler's method to solve the ODE
t, u = euler(ode_rhs, tspan, v0, dt, m, cd)

# Plot the solution
plt.plot(t, u[:, 0], '-b')
plt.show()



#General Comments
#v squared skrivs v**2
#Om man ska göra vektoroperationer, säg e.g. med resultatet av en 2:a ODE, så kan man ej använda "*/+-" utan måste använda np.multiply(a,b), np.divide(a,b),, np.add och np.subtract