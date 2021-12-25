from rk4_dynamics_robfly import RoboflyDynamics
import numpy as np
import theano.tensor as T
import time
from  visualization import TrajectoryVisualize
from ilqr.autodiff import as_function
path = "/home/airlab/Dropbox/Daksh/System_ID_project/system_identificatification_robofly/Data_interpret/"

reps= 5
dt = 0.01/reps
def tensor_constrain(u, min_bounds, max_bounds):
    """Constrains a control vector tensor variable between given bounds through
    a squashing function.

    This is implemented with Theano, so as to be auto-differentiable.

    Args:
        u: Control vector tensor variable [action_size].
        min_bounds: Minimum control bounds [action_size].
        max_bounds: Maximum control bounds [action_size].

    Returns:
        Constrained control vector tensor variable [action_size].
    """
    diff = (max_bounds - min_bounds) / 2.0
    mean = (max_bounds + min_bounds) / 2.0
    return diff * T.tanh(u) + mean


def f(x, u, i):
    # state= [theta0, theta1, theta2, omega0, omega1, omega2,-
    # x, y, z, v_x_body, v_y_body, v_z_body]
    # actions = [thrust, tau_x, tau_y]
    # state
    theta0 = x[..., 0]
    theta1 = x[..., 1]
    theta2 = x[..., 2]
    omega0 = x[..., 3]
    omega1 = x[..., 4]
    omega2 = x[..., 5]
    xpos = x[..., 6]
    ypos = x[..., 7]
    zpos = x[..., 8]
    bdot_x = x[..., 9]
    bdot_y = x[..., 10]
    bdot_z = x[..., 11]
    u0 = u[..., 0]
    u1 = u[..., 1]
    u2 = u[..., 2]
    # action = action.reshape((3,1))
    # print(action.TensorType)
    # angle wrap-around
    # B= T.stack(theta0, theta1, theta2)
    #
    # print(B.ndim)
    theta0 = (3.14159 + theta0) % (2 * 3.14159)
    theta0 = T.switch(T.lt(theta0, 0), theta0 - 2 * 3.14159, theta0)
    theta0 -= 3.14159
    theta1 = (3.14159 + theta1) % (2 * 3.14159)
    theta1 = T.switch(T.lt(theta1, 0), theta1 - 2 * 3.14159, theta1)
    theta1 -= 3.14159
    theta2 = (3.14159 + theta2) % (2 * 3.14159)
    theta2 = T.switch(T.lt(theta2, 0), theta2 - 2 * 3.14159, theta2)
    theta2 -= 3.14159

    # zero_tensor = T.as_tensor(B).zeros_like()
    #
    # temp= B - 2 * 3.14
    # B = T.switch(T.lt(B,zero_tensor),temp, B)
    # B -= 3.14159

    #dt = 1e-6
    g = 9.81
    m = 175e-6
    r_w = 0.0027
    Kp0 = 3.07e-6
    Kp1 = 1.38e-6
    Ks0 = 1.07e-6
    Ks1 = 2.55e-7
    Ks2 = 1.07e-6
    J0 = 4.5e-9
    J1 = 3.24e-9
    J2 = 2.86e-9
    neg_b_w0 = -2.5e-3
    neg_b_w4 = -0.5e-3
    neg_b_w8 = -0.8e-3

    # map input
    maxt = 1.1276
    mint = 0.83
    max_torque_roll = 18.98e-6  # +-30V
    max_torque_pitch = -9.91e-6  # +-15v
    # max_action =\
    #     np.array([max_f_l, max_torque_roll, max_torque_pitch])
    # action = tensor_constrain(action, np.asarray([0,-1,-1]), np.asarray([0,1,1]))
    # action = action * max_action  # self.scale_ctrl(action, self.min_action, self.max_action)
    u0 = (tensor_constrain(u0, 0, 1) * (maxt-mint) + mint) * m * g
    u1 = tensor_constrain(u1, -1, 1) * max_torque_roll
    u2 = tensor_constrain(u2, -1, 1) * max_torque_pitch
    s0 = T.sin(theta0)
    s1 = T.sin(theta1)
    s2 = T.sin(theta2)
    c0 = T.cos(theta0)
    c1 = T.cos(theta1)
    c2 = T.cos(theta2)
    t1 = T.tan(theta1)

    # matrix elements
    R0 = c2 * c1
    R1 = c2 * s1 * s0 - c0 * s2
    R2 = s2 * s0 + c2 * c0 * s1
    R3 = c1 * s2
    R4 = c2 * c0 + s2 * s1 * s0
    R5 = c0 * s2 * s1 - c2 * s0
    R6 = -s1
    R7 = c1 * s0
    R8 = c1 * c0

    W0 = 1
    W1 = s0 * t1
    W2 = c0 * t1
    W4 = c0
    W5 = -s0
    W7 = s0 / c1
    W8 = c0 / c1

    v_w0 = bdot_x + omega1 * r_w
    v_w1 = bdot_y - omega0 * r_w
    v_w2 = bdot_z

    fd0 = neg_b_w0 / m * v_w0
    fd1 = neg_b_w4 / m * v_w1
    fd2 = neg_b_w8 / m * v_w2

    F9 = - R2 * g + fd0 - omega1 * bdot_z + omega2 * bdot_y
    F10 = - R5 * g + fd1 - omega2 * bdot_x + omega0 * bdot_z
    F11 = u0 / m - R8 * g + fd2 - omega0 * bdot_y + omega1 * bdot_x

    F6 = R0 * bdot_x + R1 * bdot_y + R2 * bdot_z
    F7 = R3 * bdot_x + R4 * bdot_y + R5 * bdot_z
    F8 = R6 * bdot_x + R7 * bdot_y + R8 * bdot_z

    F0 = W0 * omega0 + W1 * omega1 + W2 * omega2
    F1 = W4 * omega1 + W5 * omega2
    F2 = W7 * omega1 + W8 * omega2

    F3 = (u1 - neg_b_w0 * r_w * (v_w1) - Ks0 * theta0 - Kp0 * omega0 - omega1 * J2 * omega2 + omega2 * J1 * omega1) / J0
    F4 = (u2 + neg_b_w4 * r_w * (v_w0) - Ks1 * theta1 - Kp1 * omega1 - omega2 * J0 * omega0 + omega0 * J2 * omega2) / J1
    F5 = ( - omega0 * J1 * omega1 + omega1 * J0 * omega0) / J2

    A = T.stack([theta0 + F0 * dt,
                 theta1 + F1 * dt,
                 theta2 + F2 * dt,
                 omega0 + F3 * dt,
                 omega1 + F4 * dt,
                 omega2 + F5 * dt,
                 xpos + F6 * dt,
                 ypos + F7 * dt,
                 zpos + F8 * dt,
                 bdot_x + F9 * dt,
                 bdot_y + F10 * dt,
                 bdot_z + F11 * dt,
                 ]).T
    return A

# def f(next_state, u, i):
#     time_diff = dt / 10
#
#     for j in range(10):
#         y = next_state.copy()
#         k1 = time_diff * dx(y, u, time_diff)
#         k2 = time_diff * dx(y + k1 / 2, u, time_diff)
#         k3 = time_diff * dx(y + k2 / 2, u, time_diff)
#         k4 = time_diff * dx(y + k3, u, time_diff)
#         next_state = x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
#     return next_state




traj1_state= np.load(path + "traj1_states.npy")
traj1_action= np.load(path + "traj1_action.npy")
max0 = max(traj1_action[0,:])
min0 = min(traj1_action[0,:])
max1 = max(traj1_action[1,:])
min1 = min(traj1_action[1,:])
max2 = max(traj1_action[2,:])
min2 = min(traj1_action[2,:])
# mapping input
maxis = np.asarray([max0, max1, max2]).reshape(3,1).repeat(traj1_action.shape[1],axis=1)
mins = np.asarray([min0, min1, min2]).reshape(3,1).repeat(traj1_action.shape[1],axis=1)
traj1_action[1:3,:] = (traj1_action[1:3,:]-(maxis[1:3,:]+mins[1:3,:])/2)/((maxis[1:3,:]-mins[1:3,:])/2)
traj1_action[0,:] = (traj1_action[0,:]-mins[0,:])/(maxis[0,:]-mins[0,:])
print(traj1_state.shape)
print(traj1_action.shape)

#send input to function
#roll,pitch,yaw,omega1, omega2, omega3, xpos ,ypos,zpos, xvel,yvel,zvel
x = T.vector("x")
u = T.vector("u")
i = T.scalar('i')
inputs =[x,u,i]
_tensor = f(x, u, i)
f_as = as_function(_tensor, inputs, name="f")
rollout =[]
state = traj1_state[:,0].reshape((12,))
a= time.time()
N= traj1_action.shape[1]
for i in range(N):
    for j in range(reps):
        try:
            control = traj1_action[:,i-5]
        except:
            control = np.asarray([0,0,0])
        cur_state = f_as(state, control,0)
        state= cur_state.copy()
    rollout.append(state)

print(time.time()-a)
rollout = np.asarray(rollout)
visualize = TrajectoryVisualize()
#print(rollout)
traj1_state= traj1_state.transpose()
plot_stuff = {}
plot_stuff['angles'] = np.concatenate((rollout[:,0:3], traj1_state[:,0:3]),axis=1)
plot_stuff['pos'] = np.concatenate((rollout[:,6:9], traj1_state[:,6:9]),axis=1)
plot_stuff['vel'] = np.concatenate((rollout[:,9:12], traj1_state[:,9:12]),axis=1)
plot_stuff['omega'] = np.concatenate((rollout[:,3:6], traj1_state[:,3:6]),axis=1)
visualize.plotter(plot_stuff)
visualize.show_plot()