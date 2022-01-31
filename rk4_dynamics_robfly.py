import theano.tensor as T
import theano
import theano.ifelse as ifelse
from ilqr.dynamics_parallel import AutoDiffDynamics
from ilqr.dynamics_parallel import BatchAutoDiffDynamics, tensor_constrain
import numpy as np


class RoboflyDynamics(BatchAutoDiffDynamics):

    def __init__(self, **kwargs):
        dt = 1e-4
        def dx(x, u, dt):
            # state= [theta0, theta1, theta2, omega0, omega1, omega2,-
            # x, y, z, v_x_body, v_y_body, v_z_body]
            # actions = [thrust, tau_x, tau_y]
            #state
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
            #action = action.reshape((3,1))
            #print(action.TensorType)
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
            #B -= 3.14159


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
            neg_b_w0 = -2.5e-3 / m
            neg_b_w4 = -0.5e-3 / m
            neg_b_w8 = -0.8e-3 / m

            #map input
            max_f_l= 1.1176 * m * g
            max_torque_roll = 17.64e-6 #+-30V
            max_torque_pitch = 3.31e-6 #+-15v
            # max_action =\
            #     np.array([max_f_l, max_torque_roll, max_torque_pitch])
            # action = tensor_constrain(action, np.asarray([0,-1,-1]), np.asarray([0,1,1]))
            # action = action * max_action  # self.scale_ctrl(action, self.min_action, self.max_action)
            u0 = tensor_constrain(u0, 0, 1) * max_f_l
            u1 = tensor_constrain(u1, -1, 1) * max_torque_roll
            u2 = tensor_constrain(u2, -1, 1) * max_torque_pitch
            s0 = T.sin(theta0)
            s1 = T.sin(theta1)
            s2 = T.sin(theta2)
            c0 = T.cos(theta0)
            c1 = T.cos(theta1)
            c2 = T.cos(theta2)
            t1 = T.tan(theta1)

            #matrix elements
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

            
            fd0 = neg_b_w0 * v_w0
            fd1 = neg_b_w4 * v_w1
            fd2 = neg_b_w8 * v_w2

            F9 = fd0 - omega1 * bdot_z + omega2 * bdot_y
            F10 = fd1 - omega2 * bdot_x + omega0 * bdot_z
            F11 = - u0 / m - g + fd2 - omega0 * bdot_y + omega1 * bdot_x

            F6 = R0 * bdot_x + R1 * bdot_y + R2 * bdot_z
            F7 = R3 * bdot_x + R4 * bdot_y + R5 * bdot_z
            F8 = R6 * bdot_x + R7 * bdot_y + R8 * bdot_z

            F0 = W0 * omega0 + W1 * omega1 + W2 * omega2
            F1 = W4 * omega1 + W5 * omega2
            F2 = W7 * omega1 + W8 * omega2

            F3 = (u1 - r_w * fd2 - Kp0 * theta0 - Ks0 * omega0 - omega1 * J2 * omega2 + omega2 * J1 * omega1) / J0
            F4 = (u2 + r_w * fd1 - Kp1 * theta1 - Ks1 * omega1 - omega2 * J0 * omega0 + omega0 * J2 * omega2) / J1
            F5 = (- Ks2 * omega2 - omega0 * J1 * omega1 + omega1 * J0 * omega0) / J2


            A= T.stack([F0,
                        F1,
                        F2,
                        F3,
                        F4,
                        F5,
                        F6,
                        F7,
                        F8,
                        F9,
                        F10,
                        F11]).T
            return A

        def f(x, u, i):
            k1 = dt * dx(x, u,  dt)
            k2 = dt * dx(x + k1/2, u, dt)
            k3 = dt * dx(x + k2/2, u, dt)
            k4 = dt * dx(x + k3, u, dt)
            next_state = x[..., :] + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return next_state


        super(RoboflyDynamics, self).__init__(f, state_size=12,
                                              action_size=3,
                                              **kwargs)