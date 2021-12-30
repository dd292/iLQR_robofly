import numpy as np
import matplotlib.pyplot as plt
from ilqr import iLQR
from ilqr.cost import QRCost
#from dynamics_robfly import RoboflyDynamics
from dynamics_robfly import RoboflyDynamics
import time
from  visualization import TrajectoryVisualize
from ilqr.cost import AutoDiffCost
import theano.tensor as T


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)

def hover():
    reps = 5
    dt = 0.01
    dynamics = RoboflyDynamics(dt, reps, actuator=False)
    x_goal = np.asarray([0, 0, 0, 0, 0, 0, 0.2, 0.1, 0.5, 0, 0, 0])
    Q= np.eye(dynamics.state_size)
    Q[6,6] = 1e2
    Q[7,7] = 1e2
    Q[8,8] = 1e2
    Q_terminal = 10 *  np.eye(dynamics.state_size)
    R = 0.001 * np.eye(dynamics.action_size)
    cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)
    return dynamics, cost

def act_const_hover():
    reps = 5
    dt = 0.01
    dynamics = RoboflyDynamics(dt, reps, actuator=True)

    x_goal = np.asarray([0, 0, 0, 0, 0, 0, 0.2, 0.1, 0.5, 0, 0, 0])

    #cost definition
    x_inputs = [T.dscalar("th0"), T.dscalar("th1"), T.dscalar("th2"), T.dscalar("om0"), T.dscalar("om1"),
                T.dscalar("om2"),T.dscalar("x"), T.dscalar("y"), T.dscalar("z"), T.dscalar("vbx"),
                T.dscalar("vby"),T.dscalar("vbz")]
    u_inputs = [T.dscalar("F"), T.dscalar("tau_x"), T.dscalar("tau_y")]

    x = T.stack(x_inputs)
    u = T.stack(u_inputs)
    Q = np.eye(dynamics.state_size)
    Q[6, 6] = 1e2
    Q[7, 7] = 1e2
    Q[8, 8] = 1e2
    Q_terminal = 10 * np.eye(dynamics.state_size)
    R = 0.001 * np.eye(dynamics.action_size)

    x_diff = x - x_goal
    l = x_diff.T.dot(Q).dot(x_diff) + u.T.dot(R).dot(u) #+ T.exp(-1000*x[8])/1000
    l_terminal = x_diff.T.dot(Q_terminal).dot(x_diff)

    # Compile the cost.
    # NOTE: This can be slow as it's computing and compiling the derivatives.
    # But that's okay since it's only a one-time cost on startup.
    cost = AutoDiffCost(l, l_terminal, x_inputs, u_inputs)


    #cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)
    return dynamics, cost

dynamics, cost = hover()
N = 400 # trajectory length

#define initials
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


#plotting
plot_stuff = {}
visualize = TrajectoryVisualize()
plot_stuff['angles'] = np.concatenate((xs[:,0:3], xs[:,0:3]),axis=1)
plot_stuff['pos'] = np.concatenate((xs[:,6:9], xs[:,6:9]),axis=1)
plot_stuff['vel'] = np.concatenate((xs[:,9:12], xs[:,9:12]),axis=1)
plot_stuff['omega'] = np.concatenate((xs[:,3:6], xs[:,3:6]),axis=1)
plot_stuff['input'] = us
visualize.plotter(plot_stuff)
visualize.show_plot()