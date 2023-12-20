import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def rhsODE(t,y):
    yt = np.sin(t) - y
    return yt

t0 = 0.0
t1 = 10.0
tspan = (t0, t1)
v0 = [0]

times = np.arange(t0, t1, 0.1) # Tidpunkter där lösningen lagras
vel = solve_ivp(rhsODE, tspan, v0)
print(vel) # Kolla hur utdata ser ut
print(vel.t) # Kolla tidsaxeln
# Plotta lösningen
# Tid lagras i vel.t och lösningen i vel.y .
# plottar y[0] här (enbart en ekvation här)
plt.plot(vel.t, vel.y[0], '-or')
plt.title('ODE1')
plt.xlabel('t')
plt.ylabel('y')
plt.show()
