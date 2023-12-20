import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def dSdy(y, S):
    y, v = S
    b = 0.1
    m = 2
    g = 9.81                                                
    L = 1
    return [v,
            (-b/m)*v - (g/L)*np.sin(y)]
y_0 = 0
v_0 = 3
S_0 = (y_0, v_0)

t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 100)

sol = solve_ivp(dSdy, t_span, S_0, t_eval=t_eval)

y_sol = sol.y[0]
v_sol = sol.y[1]

plt.plot(sol.t, y_sol, label='y(t)')
# plt.plot(sol.t, v_sol, label='v(t)')
plt.title('Q3: Enkelpendel')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()