import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import math
from pathlib import Path
from openweathermap.fetch_data import load_data
from typing import Tuple


FIGURES_DIR = Path(__file__).parent.joinpath('figures')


class GridFunction(object):
    def __init__(self, shape: Tuple, base_function: np.ndarray):
        """
        We define a grid function from a base function and a desired shape.
        We can manipulate the base function via downsampling, upsampling, and
        cropping to obtain a function of the desired shape.

        Parameters
        ----------
        shape: Tuple
            Desired shape of the grid function
        base_function: np.ndarray
            Base function (e.g gorilla density or temp)
        """
        self.shape = shape
        self.base_function = np.atleast_2d(base_function)

        # Current function is where we store the results of our manipulations
        self.current_function = self.base_function.copy()

    def upsample(self, current: bool = True, factors: Tuple = (2, 2)):
        """
        Upsample a function to create the desired one.

        Parameters
        ---------
        current: bool
            If true, upsampling is applied to current function, otherwise to
            base function
        factors: Tuple
            x and y factor for upsampling
        """
        if current:
            function = self.current_function
        else:
            function = self.base_function
        n, m = function.shape
        x_factor, y_factor = factors
        x = np.linspace(0, n, int(x_factor * n))
        y = np.linspace(0, m, int(y_factor * m))

        f = interp2d(range(n), range(n), function)
        upsampled = f(x, y)
        self.current_function = upsampled.reshape((x.size, y.size))

    def downsample(self, current: bool = True,
                   deltas: Tuple = (2, 2),
                   offsets: Tuple = (0, 0)):
        """
        Downsample a function to create the desired one.

        Parameters
        ---------
        current: bool
            If true, upsampling is applied to current function, otherwise to
            base function
        deltas: Tuple
            x and y factor for downsampling
        offsets: Tuple
            Coordinate where to start for downsampling
        """
        if current:
            function = self.current_function
        else:
            function = self.base_function
        dx, dy = deltas
        ox, oy = offsets
        ex, ey = function.shape
        x = np.arange(ox, ex, dx)
        y = np.arange(oy, ey, dy)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        downsampled = function[xx.flatten(), yy.flatten()]
        self.current_function = downsampled.reshape(x.size, y.size)

    def crop(self, current: bool = True,
             offsets: Tuple = (0, 0),
             shape: Tuple = None):
        """
        Crop a function to create the desired one.

        Parameters
        ---------
        current: bool
            If true, upsampling is applied to current function, otherwise to
            base function
        offsets: Tuple
            Coordinate where to start for cropping
        shape: Tuple
            Cropping shape
        """
        if current:
            function = self.current_function
        else:
            function = self.base_function
        if shape is None:
            shape = self.shape
        ox, oy = offsets
        n, m = shape
        x = np.arange(ox, ox + n)
        y = np.arange(oy, oy + m)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        cropped = function[xx.flatten(), yy.flatten()]
        self.current_function = cropped.reshape(shape)

    def fit2shape(self, shape: Tuple = None):
        """
        Applies simple sequence of transformations to guarantee current
        function has desired shape.
        """

        function = self.current_function
        if shape is None:
            shape = self.shape

        # If current function has lower dim than specified one, we upsample and crop
        if all([i <= j for i, j in zip(function.shape, shape)]):
            x_factor = math.ceil(shape[0] / function.shape[0])
            y_factor = math.ceil(shape[1] / function.shape[1])
            self.upsample(current=True, factors=(x_factor, y_factor))
            self.crop(current=True, offsets=(0, 0), shape=shape)

        # If current function has higher dim than specified one, we try to downsample and crop, otherwise just crop
        elif all([i >= j for i, j in zip(function.shape, shape)]):
            ratiox = function.shape[0] % shape[0]
            ratioy = function.shape[1] % shape[1]
            if ratiox >= 2 and ratioy >= 2:
                self.downsample(current=True, deltas=(
                    ratiox, ratioy), offsets=(0, 0))
            self.crop(current=True, offsets=(0, 0), shape=shape)
        else:
            raise NotImplementedError('We have not implemented the case where '
                                      'one dim must be upsampled and one '
                                      'downsampled')

    def plot(self, current=True, title=None):
        plt.figure()
        if current:
            function = self.current_function
        else:
            function = self.base_function
        levels = 20
        plt.contour(function, levels, linewidths=1, colors='k')
        plt.contourf(function, levels=levels)
        if title is not None:
            plt.title(title)
        plt.colorbar()

    def __call__(self, x, y):
        """
        Return function values at desired indices. If the current_function is
        not of shape self.shape, it throws an error.
        """
        assert self.current_function.shape == self.shape, \
            f'The desired shape is {self.shape}. Current function shape is ' \
            f'{self.current_function.shape} instead.'
        x = np.squeeze(np.atleast_1d(x).astype(int))
        y = np.squeeze(np.atleast_1d(y).astype(int))
        return self.current_function[x, y]


def get_gorillas_density():
    """
    Load Gorilla density from Mojmir.
    """
    gorillas = Path(__file__).parent.joinpath('gorilla_data',
                                              'gorillas_fit_model.csv')
    my_data = np.genfromtxt(gorillas, delimiter=' ')
    gorillas_rate = my_data[:, 0]
    n = int(np.sqrt(gorillas_rate.shape[0]))
    return gorillas_rate.reshape((n, n))


def get_jungle_weather(variable):
    """
    Load weather variable from stored data.
    """
    assert variable in ['temp', 'wind', 'clouds']
    data = load_data('KGS_start', n_x=10, n_y=10, delta_x=10000,
                     delta_y=10000, variables=[variable])
    return np.asarray(data[variable])


def visualize_weather_data():
    temp = GridFunction((30, 30), get_jungle_weather('temp'))
    wind = GridFunction((30, 30), get_jungle_weather('wind'))
    clouds = GridFunction((30, 30), get_jungle_weather('clouds'))

    for f, title in zip([temp, wind, clouds],
                        ['temperature', 'wind', 'clouds']):
        f.upsample(current=True, factors=(2, 2))
        # super_sampled = f(x, y)
        f.plot(current=True, title=title)
        # plot_data(super_sampled.reshape((x.size, y.size)))
        # plt.title(title)
        plt.savefig(FIGURES_DIR.joinpath(f'{title}.pdf'),
                    format='pdf', transparent=True)

    plt.show()


def main():
    density = GridFunction((30, 30), get_gorillas_density())
    density.plot(current=True)
    # density.upsample(current=True, factors=(3, 3))
    # density.plot(current=True)
    # density.crop(current=False, offsets=(40, 40), shape=(30, 30))
    # density.plot(current=True)

    constraint = GridFunction((30, 30), get_jungle_weather('clouds'))
    constraint.plot(current=True)
    constraint.fit2shape(shape=None)
    constraint.plot(current=True)

    plt.show()


if __name__ == '__main__':
    visualize_weather_data()
    main()
