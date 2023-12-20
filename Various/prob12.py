import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def oderhs(t,u):
    u1, u2 = u
    u_t = [u2,-3*u2 - 2*u1]
    return u_t


t0 = 0
t1 = 10
tspan = (0, 10)
u0 = [2, 1]
#v0=np.array([2.0, 0]) # initial values

times = np.arange(t0, t1, 0.1) # Tidpunkter där lösningen lagras
vel = solve_ivp(oderhs, tspan, u0, t_eval = times)
print(vel) # Kolla hur utdata ser ut
print(vel.t) # Kolla tidsaxeln
# Plotta lösningen
# Tid lagras i vel.t och lösningen i vel.y .
# plottar y[0] här (enbart en ekvation här)
plt.plot(vel.t, vel.y[0], label="y(t)")
plt.plot(vel.t, vel.y[1], label="y'(t)")
plt.title('ODE2')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()