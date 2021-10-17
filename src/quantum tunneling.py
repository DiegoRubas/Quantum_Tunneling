"""Author: Diego Rubas
Created on Wednesday April 7th 2020 at 4:58PM
Tested on Python 3.8
Special thanks to:
************************************************************************************************************************
*    Title: Potential Program
*    Author: Sparenberg, Jean-Marc ― Sabot, Frédéric ― Simon, Corentin
*    Date: Feb 5, 2019
*    Availability: https://github.com/jmspar/PP
************************************************************************************************************************
*    Title: Quantum Tunneling, Part 3
*    Author: Physics, Python, and Programming
*    Date: October 27, 2019
*    Availability: https://physicspython.wordpress.com/tag/potential-barrier/
************************************************************************************************************************
*    Title: Animate Line
*    Author: Seanny123
*    Date: February 21, 2017
*    Availability: https://gist.github.com/Seanny123/2c7efd90bebbe9c7bea6a1bd30a2133c#file-animate_line-py
************************************************************************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider  # , CheckButtons
from scipy import sparse as sparse
from scipy.sparse import linalg as ln


class WavePacket:

    def __init__(self, points_no, dt, sigma0=5.0, k0=1.0, x0=-50.0, x_begin=-75.0,
                 x_end=75.0, v0=0.5, vwidth=3.0):
        """This function sets up for attributes, the grid, the calculations, the visual representations and so forth."""

        """Assigns attributes"""
        self.points_no = points_no
        self.sigma0 = sigma0
        self.k0 = k0
        self.x0 = x0
        self.dt = dt
        self.prob = np.zeros(points_no)
        self.vwidth = vwidth
        self.v0 = v0

        """Initializes """
        self.time = 0

        """Discretizes space using a grid"""
        self.x, self.dx = np.linspace(x_begin, x_end, points_no, retstep=True)

        """Calculates the wave function at every point in space"""
        self.psi = (1 / ((2.0 * np.pi * sigma0 ** 2) ** 0.25))
        self.psi *= np.exp(-(self.x - x0) ** 2 / (4.0 * sigma0 ** 2)) * np.exp(1.0j * k0 * self.x)
        self.psi0 = self.psi

        """Calculates the potential at every point in space"""
        self.potential = np.array(
            [v0 if -(vwidth / 2.0) < x < (vwidth / 2.0) else 0.0 for x in self.x])

        """Calculates the evolution matrix, alleviating the mathematical load"""
        h_diag = np.ones(points_no) / self.dx ** 2 + self.potential
        h_adj = np.ones(points_no - 1) * (-0.5 / self.dx ** 2)
        hamiltonian = sparse.diags([h_adj, h_diag, h_adj], [-1, 0, 1])
        implicit = (sparse.eye(self.points_no) - dt / 2.0j * hamiltonian).tocsc()
        explicit = (sparse.eye(self.points_no) + dt / 2.0j * hamiltonian).tocsc()
        self.evolution_matrix = ln.inv(implicit).dot(explicit).tocsr()

        """Extracts the density of probability as well as the wave function from the evolve function"""
        self.density, self.wave = self.evolve()

        """Sets up the frame for graphs"""
        self.fig, self.axs = plt.subplots(3)
        self.fig.suptitle('Quantum Tunneling and Wave Packets')
        plt.subplots_adjust(0.15, 0.25, 0.95, 0.95)

        """Adds the graph of potential as a function of space"""
        self.axs[0].plot(self.x, self.potential, color='r')
        self.axs[0].set_ylim(min(-1.0, self.v0 - 0.15), max(1.0, self.v0 + 0.15))
        self.axs[0].vlines(-(vwidth / 2.0), -5, 5, colors='black', linestyles='dashed')
        self.axs[0].vlines((vwidth / 2.0), -5, 5, colors='black', linestyles='dashed')
        self.axs[0].set_ylabel('Energy')

        """Adds the graph of the wave function as a function of space"""
        self.line2, = self.axs[1].plot(self.x, np.real(self.wave))
        self.line3, = self.axs[1].plot(self.x, np.imag(self.wave))
        self.line4, = self.axs[1].plot(self.x, np.absolute(self.wave))
        self.time_text = self.axs[1].text(0.05, 0.95, '', horizontalalignment='left',
                                          verticalalignment='top', transform=self.axs[1].transAxes)
        self.axs[1].set_ylim(-0.5, 0.5)
        self.axs[1].vlines(-(vwidth / 2.0), -5, 5, colors='black', linestyles='dashed')
        self.axs[1].vlines((vwidth / 2.0), -5, 5, colors='black', linestyles='dashed')
        self.axs[1].set_ylabel('Wave\nFunction')

        """Adds the graph of the probability density as a function of space"""
        self.line1, = self.axs[2].plot(self.x, self.density, color='black')
        self.axs[2].set_ylim(-0.005, 0.035)
        self.axs[2].vlines(-(vwidth / 2.0), -5, 5, colors='black', linestyles='dashed')
        self.axs[2].vlines((vwidth / 2.0), -5, 5, colors='black', linestyles='dashed')
        self.axs[2].set_ylabel('Probability\nDensity')
        self.axs[2].set_xlabel('Position ($a_0$)')

        """Initiliazes the animation loop and its status as running"""
        self.anim = animation.FuncAnimation(self.fig, self._update, self.time_step, interval=5)
        self.anim_running = True

    def evolve(self):
        """This method calculates the state of the system one step ahead of its current.
        :return: The updated values of the probability density and the wave function as functions of space.
        """
        self.psi = self.evolution_matrix.dot(self.psi)
        self.prob = abs(self.psi) ** 2

        norm = sum(self.prob)
        self.prob /= norm
        self.psi /= norm ** 0.5

        return self.prob, self.psi

    def _pause(self, event):
        """
        This method changes the state of the animation depending on the event triggering the method.
        :param event: The event triggering the method.
        :return: None.
        """
        if self.anim_running:
            self.anim.event_source.stop()
            self.anim_running = False
        else:
            self.anim.event_source.start()
            self.anim_running = True

    def _update(self, data):
        """
        This method updates the plots of each graph. It also notifies the time slider of each time incrementation.
        :param data: The data to be plotted on each graph.
        :return: The plots of each line.
        """
        self.line1.set_ydata(self.density)
        self.line2.set_ydata(np.real(self.wave))
        self.line3.set_ydata(np.imag(self.wave))
        self.line4.set_ydata(np.absolute(self.wave))

        self.time_slider.eventson = False
        self.time_slider.set_val(self.time * 2.419e-2)
        self.time_slider.eventson = True

        return self.line1, self.line2, self.line3, self.line4

    def _update_pot(self, val):
        """
        This method notifies the rest of the program whenever the potential barrier slider is moved. It reevaluates the
        necessary blocks of code, such as the evolution matrix and the potential barrier graph. It also re-initializes
        the simulation.
        :param val: The new energy value of the potential barrier as per the potential barrier slider.
        :return: None.
        """
        self.v0 = val

        self.potential = np.array(
            [self.v0 if -(self.vwidth / 2.0) < x < (self.vwidth / 2.0) else 0.0 for x in self.x])

        h_diag = np.ones(self.points_no) / self.dx ** 2 + self.potential
        h_adj = np.ones(self.points_no - 1) * (-0.5 / self.dx ** 2)
        hamiltonian = sparse.diags([h_adj, h_diag, h_adj], [-1, 0, 1])
        implicit = (sparse.eye(self.points_no) - self.dt / 2.0j * hamiltonian).tocsc()
        explicit = (sparse.eye(self.points_no) + self.dt / 2.0j * hamiltonian).tocsc()
        self.evolution_matrix = ln.inv(implicit).dot(explicit).tocsr()

        self.axs[0].clear()
        self.axs[0].plot(self.x, self.potential, color='r')
        self.axs[0].set_ylim(min(-0.2, self.v0 - 0.15), max(0.2, self.v0 + 0.15))
        self.axs[0].vlines(-(self.vwidth / 2.0), -5, 5, colors='black', linestyles='dashed')
        self.axs[0].vlines((self.vwidth / 2.0), -5, 5, colors='black', linestyles='dashed')
        self.axs[0].set_ylabel('Energy')

        self.psi = self.psi0
        self.time = 0

    def _update_sigma(self, val):
        """
        This method notifies the rest of the program whenever the sigma naught slider is moved. It recalculates the
        entire wave function. It also re-initializes the simulation.
        :param val: The new standard deviation of the wave function as per the sigma naught slider.
        :return: None.
        """
        self.sigma0 = val

        self.psi = (1 / ((2.0 * np.pi * self.sigma0 ** 2) ** 0.25))
        self.psi *= np.exp(-(self.x - self.x0) ** 2 / (4.0 * self.sigma0 ** 2)) * np.exp(1.0j * self.k0 * self.x)
        self.psi0 = self.psi

        self.time = 0

    def _update_x(self, val):
        """
        This method notifies the rest of the program whenever the x naught slider is moved. It recalculates the
        entire wave function. It also re-initializes the simulation.
        :param val: The initial position of the wave function as per the x naught slider.
        :return: None.
        """
        self.x0 = val

        self.psi = (1 / ((2.0 * np.pi * self.sigma0 ** 2) ** 0.25))
        self.psi *= np.exp(-(self.x - self.x0) ** 2 / (4.0 * self.sigma0 ** 2)) * np.exp(1.0j * self.k0 * self.x)
        self.psi0 = self.psi

        self.time = 0

    def _reset(self, event):
        """
        This method re-initializes the simulation by resetting the time function while leaving the others unchanged.
        :param event: The event triggering this method, i.e. the reset button.
        :return: None.
        """
        self.psi = self.psi0
        self.time = 0

    def _set_val(self, val=0):
        """
        This method updates the timer when the time slider is moved.
        :param val: The desired time value.
        :return: None.
        """
        self.psi = self.psi0
        self.time = val / 2.419e-2
        for i in range(int(self.time / self.dt)):
            self.evolve()

    def time_step(self):
        """
        This method increments the timer and assigns it its new wave function and probability density for every
        position.
        :return: None.
        """
        while True:
            self.time += self.dt
            self.time_text.set_text(
                'Elapsed time: {:6.2f} fs'.format(self.time * 2.419e-2))

            self.density, self.wave = self.evolve()
            yield self.density, self.wave

    def animate(self):
        """
        This method configures the main interface consisting of the buttons and the sliders.
        :return: None.
        """
        pause_ax = self.fig.add_axes((0.7, 0.025, 0.1, 0.07))
        pause_button = Button(pause_ax, 'Play/\nPause', hovercolor='0.975')
        pause_button.on_clicked(self._pause)

        reset_ax = self.fig.add_axes((0.8, 0.025, 0.1, 0.07))
        reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
        reset_button.on_clicked(self._reset)

        slider_ax = self.fig.add_axes((0.1, 0.025, 0.5, 0.03))
        self.time_slider = Slider(slider_ax, '$t$', valmin=0, valmax=30, valinit=0.0)
        self.time_slider.on_changed(self._set_val)

        ax_pot = self.fig.add_axes([0.1, 0.065, 0.5, 0.03])
        self.pot_slider = Slider(ax_pot, '$V_0$', -1, 1, valinit=0.5)
        self.pot_slider.on_changed(self._update_pot)

        ax_sigma = self.fig.add_axes([0.1, 0.105, 0.5, 0.03])
        self.sigma_slider = Slider(ax_sigma, '$σ_0$', 0.01, 4, valinit=1)
        self.sigma_slider.on_changed(self._update_sigma)

        ax_x = self.fig.add_axes([0.1, 0.145, 0.5, 0.03])
        self.x_slider = Slider(ax_x, '$x_0$', -100, 100, valinit=-50)
        self.x_slider.on_changed(self._update_x)

        plt.show()


wave_packet = WavePacket(points_no=500, dt=1.5, vwidth=10, v0=1)
wave_packet.animate()