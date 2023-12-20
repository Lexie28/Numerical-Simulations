#y1' = y1 + y2**2 + 3x
#y2' = 3y1 + y2**3 - cos(x)
#y1(0) = 0
#y2(0) = 0

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def dSdx(x, S):
    y1, y2 = S
    return [y1 + y2**2 + 3*x,
            3*y1 + y2**3 - np.cos(x)]
y1_0 = 0
y2_0 = 0
S_0 = (y1_0, y2_0)

tspan = (0, 1)
t_eval = np.linspace(tspan[0], tspan[1], 1000)
vel = solve_ivp(dSdx, tspan, S_0, t_eval = t_eval)
print(vel) # Kolla hur utdata ser ut
print(vel.t) # Kolla tidsaxeln
# Plotta lösningen
# Tid lagras i vel.t och lösningen i vel.y .
# plottar y[0] här (enbart en ekvation här)
plt.plot(vel.t, vel.y[0], label="y1")
plt.plot(vel.t, vel.y[1], label="y2")
plt.title('Couple 1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
