import numpy as np
import matplotlib.pyplot as plt
from ilqr import iLQR
from ilqr.cost import QRCost
#from dynamics_robfly import RoboflyDynamics
from dynamics_robfly import RoboflyDynamics
import time

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)
reps= 5
dt = 0.01
dynamics = RoboflyDynamics(dt, reps)
x_goal = np.asarray([0, 0, 0, 0, 0, 0, 0.2, 0.1, 0.5, 0, 0, 0])
Q= np.eye(dynamics.state_size)
Q[6,6] = 1e2
Q[7,7] = 1e2
Q[8,8] = 1e2
Q_terminal = 10 *  np.eye(dynamics.state_size)
R = 0.001 * np.eye(dynamics.action_size)
cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)

N = 400
x0 = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
np.random.seed(1234)
u0_init = 0 * np.random.uniform(0, 1, (N, 1))
u1_init = 0 * np.random.uniform(-1, 1, (N, 2))
us_init = np.concatenate((u0_init, u1_init), axis=1)

ilqr = iLQR(dynamics, cost, N)
J_hist = []
start= time.time()
xs, us = ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)
print(time.time()-start)
