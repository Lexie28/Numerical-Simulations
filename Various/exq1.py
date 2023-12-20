import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
# Definiera högerledet hos ODEn
def ode_rhs(t,y):
    C = 1.8
    B= 10000
    yt = C*y*(1-(y/B))
    return yt

t0 = 0.0 # Starttid
t1 = 8.0 # sluttid
tspan = (t0, t1) # Tidsintervall
y0 = [10] # Begynnelsevärde, array
# Definiera parametrar

times = np.arange(t0, t1, 0.1)
vel = solve_ivp(ode_rhs, tspan, y0, t_eval=times)
print(vel)
print(vel.t)
plt.plot(vel.t, vel.y[0])
plt.title('Q1: Bakterietillväxt')
plt.xlabel('t (sekunder)')
plt.ylabel('Bakterietillväxt')
plt.show()