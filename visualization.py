import matplotlib.pyplot as plt
import numpy as np
class TrajectoryVisualize():
    def __init__(self):
       self.figure_number=0

    def plot_angles(self, fig, axs, roll,pitch,yaw, rollT, pitchT, yawT):
        X = np.linspace(0, roll.shape[0] / 100, roll.shape[0])
        axs[0].plot(X, roll, 'b')
        axs[1].plot(X, pitch, 'g')
        axs[0].plot(X, rollT, '--b')
        axs[1].plot(X, pitchT, '--g')
        #plt.plot(X, yaw)
        axs[0].legend(('Roll', 'Roll_True'))#, 'Yaw'))
        axs[1].legend(('Pitch', 'pitch_True'))
        #plt.ylim(top=1, bottom=-1)


    def plot_positions(self, fig, axs,  xpos, ypos, zpos, x_t, y_t, z_t):
        X = np.linspace(0, xpos.shape[0]/100, xpos.shape[0])
        axs[0].plot(X, xpos, 'b')
        axs[1].plot(X, ypos, 'g')
        axs[2].plot(X, zpos, 'k')
        axs[0].plot(X, x_t, '--b')
        axs[1].plot(X, y_t, '--g')
        axs[2].plot(X, z_t, '--k')
        axs[0].legend(('X', 'X_true'))
        axs[1].legend(('Y', 'Y_true'))
        axs[2].legend(('Z', 'Z_true'))
        #plt.ylim(top=1, bottom=-1)


    def plot_omegas(self, fig, axs, omega0, omega1, omega2, omega0_t, omega1_t, omega2_t):
        X = np.linspace(0, omega0.shape[0]/100, omega0.shape[0])
        axs[0].plot(X, omega0, 'b')
        axs[1].plot(X, omega1, 'g')
        axs[0].plot(X, omega0_t, '--b')
        axs[1].plot(X, omega1_t, '--g')
        axs[0].legend(('omega0', 'true_omega0'))
        axs[1].legend(('omega1', 'true_omega1'))

        #plt.legend(('omega0', 'omega1', 'om0_true', 'om1_true'))
        #plt.ylim(top=1, bottom=-1)


    def plot_vel(self, fig, axs, xvel, yvel, zvel, x_t, y_t, z_t):
        X = np.linspace(0, zvel.shape[0] / 100, zvel.shape[0])
        axs[0].plot(X, xvel, 'b')
        axs[1].plot(X, yvel, 'g')
        axs[2].plot(X, zvel, 'k')
        axs[0].plot(X, x_t, '--b')
        axs[1].plot(X, y_t, '--g')
        axs[2].plot(X, z_t, '--k')
        axs[0].legend(('Xvel', 'X_true'))
        axs[1].legend(('Yvel', 'Y_true'))
        axs[2].legend(('Zvel', 'Z_true'))
        #plt.legend(('X_velocity', 'Y_velocity', 'Z_velocity', 'True_x_vel', 'True_y_vel', 'True_z_vel'))
        #plt.ylim(top=1, bottom=-1)

    def plot_input(self, fig, axs, u0, u1, u2):
        X = np.linspace(0, u0.shape[0] / 100, u0.shape[0])
        axs[0].plot(X, self.map_controls(u0, 50, 200), 'b')# map 50 to 200
        axs[1].plot(X, self.map_controls(u1, -20, 20), 'g')# map -20 to 20
        axs[2].plot(X, self.map_controls(u2, -30, 30), 'k')# map -30 to 30

        axs[0].legend('thrust')
        axs[1].legend('torque_x')
        axs[2].legend('torque_y')
    def map_controls(self, input, min_bounds, max_bounds):
        diff = (max_bounds - min_bounds) / 2.0
        mean = (max_bounds + min_bounds) / 2.0
        return diff * np.tanh(input) + mean

    def plotter(self, dict):
        for key,value in dict.items():
            fig, axs= plt.subplots(3)
            if key=='angles':
                self.plot_angles(fig, axs, value[:,0],value[:,1],value[:,2],value[:,3],value[:,4],value[:,5])
            elif key=='pos':
                self.plot_positions(fig, axs, value[:,0],value[:,1],value[:,2],value[:,3],value[:,4],value[:,5])
            elif key == 'vel':
                self.plot_vel(fig, axs, value[:,0],value[:,1],value[:,2],value[:,3],value[:,4],value[:,5])
            elif key == 'omega':
                self.plot_omegas(fig, axs, value[:,0],value[:,1],value[:,2],value[:,3],value[:,4],value[:,5])
            elif key == 'input':
                self.plot_input(fig, axs, value[:, 0], value[:, 1], value[:, 2])
            self.figure_number+=1

    def show_plot(self):
        plt.show()
