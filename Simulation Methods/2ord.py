# We get x.. = -x.**2 + sin(x)
# Take x (what we want to solve for). Then define x.=v so that v is a new variable
# Note that x. = v is one differential equation
# Since v. = x.. = -x.**2 + sin(x) we get another differential
# These are two coupled ODE1s. 
#Our equations:
# x. = v
# v. = -v**2 + sin(x)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def dSdx(x,S):
    x, v = S
    return [v,
            -v**2 + np.sin(x)]
x_0 = 0 #x and v require initial conditions
v_0 = 5
S_0 = (x_0, v_0)

t_span = (0, 1)
t_eval = np.linspace(t_span[0], t_span[1], 100)

sol = solve_ivp(dSdx, t_span, S_0, t_eval=t_eval)

x_sol = sol.y[0]
v_sol = sol.y[1]

plt.plot(sol.t, x_sol, label='x(t)')
plt.plot(sol.t, v_sol, label='v(t)')
plt.title('Coupled 2nd Order Differential Equations')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()


##tspan = (0, 1)
##t_eval = np.linspace(tspan[0], tspan[1], 1000)
##vel = solve_ivp(dSdx, tspan, S_0, t_eval = t_eval)
##print(vel) # Kolla hur utdata ser ut
##print(vel.t) # Kolla tidsaxeln
# Plotta lösningen
# Tid lagras i vel.t och lösningen i vel.y .
# plottar y[0] här (enbart en ekvation här)
##plt.plot(vel.t, vel.y[0], label="x(t)")
##plt.plot(vel.t, vel.y[1], label="v(t)")
##plt.title('Couple 2nd order')
##plt.xlabel('t')
##plt.ylabel('x')
##plt.legend()
##plt.show()