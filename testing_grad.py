from ilqr.autodiff import (as_function, batch_jacobian, hessian_vector,
                       jacobian_vector)
import numpy as np

import theano.tensor as T
import matplotlib.pyplot as plt
import theano

def analytical_son(t):
    return np.exp(-t) + np.exp(-2*t)

def rk4(f, x,  y, h):

        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)
        y += (k1 + (2 * k2) + (2 * k3) + k4) / 6

        return y
def func (t, x):
    return np.asarray([x[1], -2 * x[0] - 3 * x[1]])

t0 = 0
tf = 10
h = 0.01
T_span = np.linspace(0, tf, int(tf/0.01))
x0 = np.asarray([2.0 , -3.0])
z = x0
output=[]
for iternum, i in enumerate(T_span):

    z = rk4(func, i, z, h)
    output.append(z.copy())
output= np.asarray(output)

ana_sol= analytical_son(T_span)

plt.plot(T_span, output[:,0], label='RK4')
plt.plot(T_span, ana_sol, 'r--', label='analytical')
plt.xlabel('t')
plt.ylabel('dot_x')
plt.legend()
#plt.show()
dt=0.01
def f(x):
    k1 = dt * symb_func(x)
    k2 = dt * symb_func(x + k1 / 2)
    k3 = dt * symb_func(x + k2 / 2)
    k4 = dt * symb_func(x + k3)
    next_state = x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state

def symb_func(x):
    dt=0.01
    x0 = x[... ,0]
    x1 = x[... ,1]
    return T.stack([x0 + x1*dt, x1 + (-2 * x0 - 3 * x1**3)*dt]).T
#x0= T.dvector('x0')
#x1= T.dscalar('x1')
x= T.dvector('x')
#u = T.dvector("u")
#i = T.dscalar("i")
#J= jacobian_vector(T.stack([x0, -2 * x0 - 3 * x1**3]),[x0,x1],2)
inputs = [x]
J = batch_jacobian(symb_func, inputs, 2)
f_x = as_function(J, inputs, name="f_x")
print(f_x)
#f = theano.function([x0,x1],J)
#A= np.array([5,10],[6,11])
#print(f(A))
print(f_x([1,1]))