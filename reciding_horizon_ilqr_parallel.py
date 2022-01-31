from testing_noisy_dynamics import f
import theano.tensor as Tensor
import numpy as np
from ilqr.controller_tessa_parallel import iLQR
from ilqr.batch_cost import BatchQRCost
from dynamics_robfly_4inputs import RoboflyDynamics
import time
from single_graph_visualization import TrajectoryVisualize as TV
from ilqr.autodiff import as_function
from tqdm import tqdm
import copy

def hover(boxQP,dt):
    dynamics = RoboflyDynamics(dt, reps, boxQP, True)
    x_goal = np.asarray([0, 0, 0, 0, 0, 0, 0.2, 0.1, 0.5, 0, 0, 0])
    Q= np.eye(dynamics.state_size)
    Q[6,6] = 1e2
    Q[7,7] = 1e2
    Q[8,8] = 1e2
    Q_terminal = 100 * np.eye(dynamics.state_size)
    #Q_terminal[8,8] = 1e4
    R = 0.001 * np.eye(dynamics.action_size)
    R[0,0] = 1e-5
    cost = BatchQRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)
    return dynamics, cost

box_QP = True


#define initials
x0 = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((12,1))
np.random.seed(1234)
T = 0.5 #horizon length
traj_length = 4 #trajectory length
reps = 5 # integrations per timestep
dt = 0.01 # timestep
steps = 1
dynamics, cost = hover(box_QP, dt)
N= int(T / dt) # horizon steps
traj_steps = int(traj_length/dt)
u0_init = 100 * np.ones((1, N))
u1_init = 0 * np.random.uniform(-1, 1, (2, N))
us_init = np.concatenate((u0_init, u1_init), axis=0)

##controller
state_traj = np.zeros((x0.shape[0], traj_steps))
action_traj = np.zeros((us_init.shape[0], traj_steps))

## Noisy system
x = Tensor.dmatrix("x")
u = Tensor.dmatrix("u")
i = Tensor.scalar('i')
inputs = [x,u,i]
_tensor = f(x, u, i)
f_as = as_function(_tensor, inputs, name="f")

# constraints
start= time.time()
bias = 200
if box_QP:
    ub = np.array([bias, 20, 30]).reshape((3, 1))
    lb = np.array([50, -20, -30]).reshape((3, 1))
    limits = np.concatenate((lb, ub), axis=1)
else:
    limits = None  # np.concatenate((lb,ub),axis=1)
# simulation
for i in tqdm(range(int(traj_steps/steps))):
    if not i:
        x0 = x0.reshape((12,1))
    else:
        x0 = xout.reshape((12,1))
    ilqr = iLQR(dynamics, cost)
    xs, us, cost_fit = ilqr.fit(x0, us_init, limits)
    xin = x0.copy()
    #for k in range(steps):
    for j in range(reps):
        xin = xin.reshape((1,12))
        xout = f_as(xin, us[:, 0].reshape((1,3)), 0)
        xin = xout.copy()
    action_traj[:, i] = us[:, 0]
    state_traj[:, i] = xout.reshape((12,))
    us_init = us.copy()

print(time.time()-start)
#plotting
state_traj= state_traj.transpose()
action_traj = action_traj.transpose()
plot_stuff = {}
visualize = TV()
plot_stuff['angles'] = np.concatenate((state_traj[:,0:3], state_traj[:,0:3]),axis=1)
plot_stuff['pos'] = np.concatenate((state_traj[:,6:9], state_traj[:,6:9]),axis=1)
plot_stuff['vel'] = np.concatenate((state_traj[:,9:12], state_traj[:,9:12]),axis=1)
plot_stuff['omega'] = np.concatenate((state_traj[:,3:6], state_traj[:,3:6]),axis=1)
plot_stuff['input'] = action_traj
visualize.plotter(plot_stuff,box_QP)
visualize.show_plot()

