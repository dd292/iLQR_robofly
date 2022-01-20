import theano.tensor as T
import theano
import theano.ifelse as ifelse
from ilqr.dynamics import AutoDiffDynamics
from ilqr.dynamics import BatchAutoDiffDynamics, tensor_constrain
import numpy as np


class RoboflyDynamics(BatchAutoDiffDynamics):

    def __init__(self, dt, reps, box_QP, actuator,  **kwargs):

        def dx(x, u, time_step):
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

            # dt = 1e-6
            g = 9.81
            m = 175e-6
            r_w = 0.0027
            Kp0 = 3.07e-6
            Kp1 = 1.38e-6
            Ks0 = 1.07e-6
            Ks1 = 2.55e-7

            J0 = 4.5e-9
            J1 = 3.24e-9
            J2 = 2.86e-9
            neg_b_w0 = -2.5e-3
            neg_b_w4 = -0.5e-3
            neg_b_w8 = -0.8e-3
            bias = 180
            u0 = u0 - T.abs_(u1) - T.abs_(u2)
            # map input
            if actuator:
                if not box_QP:
                    #squash constraints for actuator
                    u0 = tensor_constrain(u0, 50, 200)  # map 50 to 200
                    u1 = tensor_constrain(u1, -20, 20)  # map -20 to 20
                    u2 = tensor_constrain(u2, -30, 30)  # map -30 to 30
                #u0 = ((2 * u0) * 1.086 - 2 * 110.31 ) * 1e-6 * g
                u0 = u0 * 1e-6 *g
                u1 = (u1 * 0.94) * 1e-6
                u2 = (u2 * 0.33) * 1e-6
            else:
                maxt = 1.1276
                mint = 0.83
                max_torque_roll = 18.98e-6  # +-30V
                max_torque_pitch = -9.91e-6  # +-15v
                u0 = (tensor_constrain(u0, 0, 1) * (maxt - mint) + mint) * m * g
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

            F3 = (u1 - neg_b_w0 * r_w * (
                v_w1) - Ks0 * theta0 - Kp0 * omega0 - omega1 * J2 * omega2 + omega2 * J1 * omega1) / J0
            F4 = (u2 + neg_b_w4 * r_w * (
                v_w0) - Ks1 * theta1 - Kp1 * omega1 - omega2 * J0 * omega0 + omega0 * J2 * omega2) / J1
            F5 = (- omega0 * J1 * omega1 + omega1 * J0 * omega0) / J2

            A = T.stack([theta0 + F0 * time_step,
                         theta1 + F1 * time_step,
                         theta2 + F2 * time_step,
                         omega0 + F3 * time_step,
                         omega1 + F4 * time_step,
                         omega2 + F5 * time_step,
                         xpos + F6 * time_step,
                         ypos + F7 * time_step,
                         zpos + F8 * time_step,
                         bdot_x + F9 * time_step,
                         bdot_y + F10 * time_step,
                         bdot_z + F11 * time_step,
                         ]).T
            return A
        def f(x, u, i):
            for i in range(reps):
                x = dx(x, u, dt/reps)
            return x
        super(RoboflyDynamics, self).__init__(f, state_size=12,
                                              action_size=3,
                                              **kwargs)