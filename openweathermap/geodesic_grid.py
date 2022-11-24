import geopy
import geopy.distance
from typing import Iterable, Tuple


__all__ = ['create_grid']


def create_grid(start: geopy.Point,
                delta_x: float,
                delta_y: float,
                n_x: int,
                n_y: int) -> Tuple[Iterable[geopy.Point],
                                   Iterable[list], Iterable[list]]:
    """
    Compute the geodesic grid.

    We compute a grid of coordinates from a starting coordinate and then taking
    n_x steps of width delta_x (in meters) to the East and n_y steps of width
    delta_y to the North.

    Parameters
    ----------
    start: geopy.Point
    delta_x: float
    delta_y: float
    n_x: int
    n_y: int

    Retuns
    ------
    coordinates: list[geopy.Point]
    positions: list[[float, float]]
    ind: list[[int, int]]

    """
    coordinates = []
    positions = []
    indices = []

    for i in range(n_x):
        d_x = geopy.distance.distance(kilometers=float(i * delta_x * 1e-3))

        # Bearing 90 corresponds to moving East
        intermediate = d_x.destination(point=start, bearing=90)
        for j in range(n_y):
            d_y = geopy.distance.distance(kilometers=float(j * delta_y * 1e-3))

            # Bearing 0 corresponds to moving North
            end = d_y.destination(point=intermediate, bearing=0)
            coordinates.append(end)
            positions.append([d_x.m, d_y.m])
            indices.append([i, j])

    return coordinates, positions, indices


if __name__ == '__main__':
    start = geopy.Point(48.853, 2.349)
    coord, pos, ind = create_grid(start, 500, 500, 2, 2)
    for c, p, i in zip(coord, pos, ind):
        print(f'Coordinate {c}\n'
              f'Distance from start {p[0]}m E {p[1]}m N\n'
              f'Total distance from start {geopy.distance.distance(start, c).m}m\n'
              f'Indices {i}')
