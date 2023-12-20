import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.82  # Gravity (m/s^2)
m = 70    # Mass of the jumper (kg)
k = 30    # Spring constant (N/m)
c = 2     # Damping coefficient (Ns/m)
rope_length = 20  # Length of the rope (m)
initial_height = 100  # Initial height (m)
terminal_velocity = np.sqrt(m * g / c)  # Terminal velocity (m/s)

def ode_r(t, y):
    position, velocity = y
    if position <= rope_length:
        # Jumper is above or at the rope, and the cord is slack
        acceleration = g
    else:
        # Jumper is below the rope, and the cord is taut
        tension = k * (rope_length - position)  # Tension in the rope
        damping = c * velocity  # Damping force
        net_force = m * g - tension - damping
        acceleration = net_force / m
        acceleration = max(acceleration, -g)  # Ensure negative acceleration doesn't exceed free fall

    return [velocity, acceleration]

# Time span and initial conditions
tspan = [0, 200]
y0 = [initial_height, 0]

# Solve the differential equation
from scipy.integrate import solve_ivp
sol = solve_ivp(ode_r, tspan, y0, t_eval=np.linspace(tspan[0], tspan[1], 1000))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Height')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.title('Bungee Jumper Simulation')
plt.grid(True)
plt.legend()
plt.show()
