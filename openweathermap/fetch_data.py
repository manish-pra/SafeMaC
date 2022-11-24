import json
import geopy
from openweathermap.geodesic_grid import create_grid
from openweathermap.owm import get_own_data
from pathlib import Path
import time
from typing import List
import matplotlib.pyplot as plt


WEATHER_DATA_DIR = Path(__file__).parent.joinpath('weather_data')
ANTARCTIC_LOCATIONS = ['McMurdo', 'Dumont Urville', 'BelgranoII']
KGS_LOCATIONS = ['KGS_start', 'KGS_end']
workspace = "openweathermap/"


def get_data(location: str,
             delta_x: float,
             delta_y: float,
             n_x: int,
             n_y: int):
    """
    Get weather data for locations specified with geodesic grid starting from one of the base location.

    The antarctic_locations.json file contains location of some research
    stations in antarctica. We compute geodesic grid of coordinates starting
    from those and moving n_x times by delta_x to the East and n_y times by
    delta_y to the North (see geodesic_grid).
    """
    # Location must be one of those for which we have coordinates

    if location in ANTARCTIC_LOCATIONS:
        fname = workspace + 'antarctic_locations.json'
    elif location in KGS_LOCATIONS:
        fname = workspace + 'kagwene_gorilla_sanctuary_location.json'
    else:
        raise (
            ValueError, f'Location must be in {ANTARCTIC_LOCATIONS + KGS_LOCATIONS}. Got {location} instead.')

    with open(fname, 'r') as handle:
        locations = json.load(handle)
        lat, lon = locations[location]

    # Compute grid of coordinates
    start = geopy.Point(latitude=lat, longitude=lon)
    coordinates, positions, indices = create_grid(start,
                                                  delta_x=delta_x,
                                                  delta_y=delta_y,
                                                  n_x=n_x,
                                                  n_y=n_y)

    # Initialize the data dict with some basic info
    data = dict(location=location, delta_x=delta_x, delta_y=delta_y, n_x=n_x,
                n_y=n_y)

    # Query data from OWM for all coordinates
    for c, p, i in zip(coordinates, positions, indices):
        current_data = get_own_data(c.latitude, c.longitude)
        current_data.update({'distance_from_base_location': p})

        data.update({f'{location}-{i[0]}_{i[1]}':  current_data})
        # Can only make 60 queries per minute
        time.sleep(1)

    # Store the data
    WEATHER_DATA_DIR.mkdir(exist_ok=True)
    fname = get_fname(location, delta_x, delta_y, n_x, n_y, 'json')

    with open(WEATHER_DATA_DIR.joinpath(fname), 'w') as handle:
        json.dump(data, handle)


def get_fname(location: str,
              delta_x: float,
              delta_y: float,
              n_x: int,
              n_y: int,
              extension: str = 'json'):
    """
    Helper function to standardize file names.
    """
    return f'{location}-dx_{delta_x}-dy_{delta_y}-nx_{n_x}-ny_{n_y}.{extension}'


def load_data(location: str,
              delta_x: float,
              delta_y: float,
              n_x: int,
              n_y: int,
              variables: List[str] = ['temp']):
    """
    Load data stored by get_data.

    All parameters are standard except for the list of variables, which allows
    us to specify which quantities (e.g. temperature or wind) we should load.
    """
    fname = get_fname(location, delta_x, delta_y, n_x, n_y, 'json')
    with open(WEATHER_DATA_DIR.joinpath(fname), 'r') as handle:
        raw_data = json.load(handle)

    n_x = raw_data['n_x']
    n_y = raw_data['n_y']
    location = raw_data['location']

    data = dict()

    variables.append('indices')

    for v in variables:
        v_value = []
        for i in range(n_x):
            tmp_l = []
            for j in range(n_y):
                if v == 'indices':
                    tmp_l.append([i, j])
                else:
                    key = f'{location}-{i}_{j}'
                    if v == 'temp':
                        tmp_l.append(raw_data[key]['main']['temp'])
                    elif v == 'wind':
                        tmp_l.append(raw_data[key]['wind']['speed'])
                    elif v == 'clouds':
                        tmp_l.append(raw_data[key]['clouds']['all'])
            v_value.append(tmp_l)
        data.update({v: v_value})
    return data


def plot_data(value):
    plt.figure()
    levels = 20
    plt.contour(value, levels, linewidths=1, colors='k')
    plt.contourf(value, levels=levels)
    plt.colorbar()


def main(location='McMurdo', delta_x=2000, delta_y=2000, n_x=10, n_y=10):
    fname = get_fname(location, delta_x, delta_y, n_x, n_y, 'json')

    # Get data from OWM if we do not have it already
    if not WEATHER_DATA_DIR.joinpath(fname).is_file():
        get_data(location=location, delta_x=delta_x,
                 delta_y=delta_y, n_x=n_x, n_y=n_y)

    # Load data
    data = load_data(location=location, delta_x=delta_x, delta_y=delta_y,
                     n_x=n_x, n_y=n_y,
                     variables=['temp', 'wind', 'clouds'])

    # Plots
    plot_data(data['temp'])
    plot_data(data['wind'])
    plot_data(data['clouds'])


if __name__ == '__main__':
    main(location='KGS_end', n_x=10, n_y=10, delta_x=10000, delta_y=10000)
