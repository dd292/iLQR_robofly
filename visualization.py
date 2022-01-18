import matplotlib.pyplot as plt
import numpy as np
class TrajectoryVisualize():
    def __init__(self):
       self.figure_number=0

    def plot_angles(self, fig, axs, roll,pitch,yaw, rollT, pitchT, yawT, comparison= False):
        legend0 = []
        legend1 = []
        X = np.linspace(0, roll.shape[0] / 100, roll.shape[0])
        axs[0].plot(X, roll, 'b')
        axs[1].plot(X, pitch, 'g')
        legend0.append('Roll (rad)')
        legend1.append('pitch (rad)')
        if comparison:
            axs[0].plot(X, rollT, '--b')
            axs[1].plot(X, pitchT, '--g')
            legend0.append('Roll_True')
            legend1.append('pitch_True')

        #plt.plot(X, yaw)
        axs[0].legend(legend0)#, 'Yaw'))
        axs[1].legend(legend1)
        #plt.ylim(top=1, bottom=-1)
        plt.savefig('results/angles.png')

    def plot_positions(self, fig, axs,  xpos, ypos, zpos, x_t, y_t, z_t, comparison= False):
        legend0 = []
        legend1 = []
        legend2 = []
        X = np.linspace(0, xpos.shape[0]/100, xpos.shape[0])
        axs[0].plot(X, xpos, 'b')
        axs[1].plot(X, ypos, 'g')
        axs[2].plot(X, zpos, 'k')
        legend0.append('X pos (m)')
        legend1.append('Y pos (m)')
        legend2.append('Z pos (m)')
        if comparison:
            axs[0].plot(X, x_t, '--b')
            axs[1].plot(X, y_t, '--g')
            axs[2].plot(X, z_t, '--k')
            legend0.append('X_true')
            legend1.append('Y_true')
            legend2.append('Z_true')

        axs[0].legend(legend0)
        axs[1].legend(legend1)
        axs[2].legend(legend2)
        #plt.ylim(top=1, bottom=-1)
        plt.savefig('results/positions.png')

    def plot_omegas(self, fig, axs, omega0, omega1, omega2, omega0_t, omega1_t, omega2_t, comparison= False):
        legend0 = []
        legend1 = []
        X = np.linspace(0, omega0.shape[0]/100, omega0.shape[0])
        axs[0].plot(X, omega0, 'b')
        axs[1].plot(X, omega1, 'g')
        legend0.append('omega0 (rad/sec)')
        legend1.append('omega1 (rad/sec)')
        if comparison:
            axs[0].plot(X, omega0_t, '--b')
            axs[1].plot(X, omega1_t, '--g')
            legend0.append('true_omega0')
            legend1.append('true_omega1')
        axs[0].legend(legend0)
        axs[1].legend(legend1)

        #plt.legend(('omega0', 'omega1', 'om0_true', 'om1_true'))
        #plt.ylim(top=1, bottom=-1)
        plt.savefig('results/omegas.png')


    def plot_vel(self, fig, axs, xvel, yvel, zvel, x_t, y_t, z_t, comparison= False):
        legend0 = []
        legend1 = []
        legend2 = []
        X = np.linspace(0, zvel.shape[0] / 100, zvel.shape[0])
        axs[0].plot(X, xvel, 'b')
        axs[1].plot(X, yvel, 'g')
        axs[2].plot(X, zvel, 'k')
        axs[0].plot(X, x_t, '--b')
        axs[1].plot(X, y_t, '--g')
        axs[2].plot(X, z_t, '--k')
        legend0.append('Xvel (m/sec)')
        legend1.append('Yvel (m/sec)')
        legend2.append('Zvel (m/sec)')
        if comparison:
            legend0.append('X_true')
            legend1.append('Y_true')
            legend2.append('Z_true')
        axs[0].legend(legend0)
        axs[1].legend(legend1)
        axs[2].legend(legend2)
        #plt.legend(('X_velocity', 'Y_velocity', 'Z_velocity', 'True_x_vel', 'True_y_vel', 'True_z_vel'))
        #plt.ylim(top=1, bottom=-1)
        plt.savefig('results/Lin_velocity.png')

    def plot_input(self, fig, box_qp , axs, u0, u1, u2, comparison= False):
        X = np.linspace(0, u0.shape[0] / 100, u0.shape[0])
        if box_qp:
            axs[0].plot(X, u0, 'b')  # map 50 to 200
            axs[1].plot(X, u1, 'g')  # map -20 to 20
            axs[2].plot(X, u2, 'k')  # map -30 to 30

        else:
            axs[0].plot(X, self.map_controls(u0, 50, 200), 'b')# map 50 to 200
            axs[1].plot(X, self.map_controls(u1, -20, 20), 'g')# map -20 to 20
            axs[2].plot(X, self.map_controls(u2, -30, 30), 'k')# map -30 to 30

        axs[0].legend(['Base_Amp (V)'])
        axs[1].legend(['Delta_Amp (V)'])
        axs[2].legend(['Offset (V)'])
        plt.savefig('results/actions.png')
    def map_controls(self, input, min_bounds, max_bounds):
        diff = (max_bounds - min_bounds) / 2.0
        mean = (max_bounds + min_bounds) / 2.0
        return diff * np.tanh(input) + mean

    def plotter(self, dict, box_qp = False ):
        comparison = False
        for key,value in dict.items():
            fig, axs= plt.subplots(3)
            if key=='angles':
                self.plot_angles(fig, axs, value[:,0],value[:,1],value[:,2],value[:,3],value[:,4],value[:,5], comparison)
            elif key=='pos':
                self.plot_positions(fig, axs, value[:,0],value[:,1],value[:,2],value[:,3],value[:,4],value[:,5], comparison)
            elif key == 'vel':
                self.plot_vel(fig, axs, value[:,0],value[:,1],value[:,2],value[:,3],value[:,4],value[:,5], comparison)
            elif key == 'omega':
                self.plot_omegas(fig, axs, value[:,0],value[:,1],value[:,2],value[:,3],value[:,4],value[:,5], comparison)
            elif key == 'input':
                self.plot_input(fig, box_qp, axs, value[:, 0], value[:, 1], value[:, 2], comparison)
            self.figure_number+=1

    def show_plot(self):

        plt.show()
