import numpy as np
import matplotlib.pyplot as plt
from ilqr.controller_tessa import iLQR
from ilqr.cost import QRCost
#from dynamics_robfly import RoboflyDynamics
from dynamics_robfly_4inputs import RoboflyDynamics
import time
from  visualization import TrajectoryVisualize
from ilqr.cost import AutoDiffCost
import theano.tensor as T


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)

def hover(boxQP):
    reps = 5
    dt = 0.01
    dynamics = RoboflyDynamics(dt, reps, boxQP, actuator = True)
    x_goal = np.asarray([0, 0, 0, 0, 0, 0, 0.2, 0.1, 0.5, 0, 0, 0])
    Q= np.eye(dynamics.state_size)
    Q[6,6] = 1e1
    Q[7,7] = 1e1
    Q[8,8] = 1e1
    Q_terminal = 100 * np.eye(dynamics.state_size)
    #Q_terminal[8,8] = 1e4
    R = 0.001 * np.eye(dynamics.action_size)
    R[0,0] = 1e-5
    cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)
    return dynamics, cost

def act_const_hover(box_QP):
    reps = 5
    dt = 0.01
    dynamics = RoboflyDynamics(dt, reps, box_QP, actuator = True)

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

box_QP = True
dynamics, cost = hover(box_QP)
N = 400 # trajectory length

#define initials
x0 = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((12,1))
np.random.seed(1234)
u0_init = 100 * np.ones((1,N)) # * np.random.uniform(0, 1, (N, 1))
u1_init = 0 * np.random.uniform(-1, 1, (2, N))
us_init = np.concatenate((u0_init, u1_init), axis=0)
bias = 150
if box_QP:
    ub = np.array([bias, 20, 30]).reshape((3,1))
    lb = np.array([50, -20, -30]).reshape((3,1))
    limits = np.concatenate((lb,ub),axis=1)
else:
    limits= None# np.concatenate((lb,ub),axis=1)
ilqr = iLQR(dynamics, cost)
J_hist = []
start= time.time()
xs, us, cost = ilqr.fit(x0, us_init, limits)
print(time.time()-start)
xs= xs.transpose()
us= us.transpose()


#plotting
plot_stuff = {}
visualize = TrajectoryVisualize()
plot_stuff['angles'] = np.concatenate((xs[:,0:3], xs[:,0:3]),axis=1)
plot_stuff['pos'] = np.concatenate((xs[:,6:9], xs[:,6:9]),axis=1)
plot_stuff['vel'] = np.concatenate((xs[:,9:12], xs[:,9:12]),axis=1)
plot_stuff['omega'] = np.concatenate((xs[:,3:6], xs[:,3:6]),axis=1)
plot_stuff['input'] = us
visualize.plotter(plot_stuff,box_QP)
visualize.show_plot()

