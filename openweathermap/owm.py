import requests
import json
import geopy
from pathlib import Path


__all__ = ['get_own_data']


def get_own_data(lat: float, lon: float) -> dict:
    """
    Get Open Weather Map data for specific latitude and longitude

    Parameters
    ----------
    lat: float
        Latitude
    lon: float
        Longitude

    Returns
    -------
    data: dict
        Dictionary of weather data in format like json here: https://openweathermap.org/current
    """
    # Get API key
    try:
        with open('APIKey.txt', 'r') as handle:
            api_key = handle.readline()
    except FileNotFoundError:
        print('You must specify your Open Weather Map API key in APIkey.txt')

    # Compute url for API query
    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    url = f'{base_url}lat={lat}&lon={lon}&appid={api_key}'

    print(f'Querying {lat} {lon}...')

    response = requests.get(url)
    data = response.json()
    return data


def example():
    """
    Example to get and store weather in Milan.
    """
    Milan = geopy.Point(latitude=45.4642162, longitude=9.1898182)
    weather_data = get_own_data(lat=Milan.latitude, lon=Milan.longitude)
    print(weather_data)

    path = Path(__file__).parent.joinpath('weather_data')
    path.mkdir(exist_ok=True)

    with open(path.joinpath('milan_weather.json'), 'w') as handle:
        json.dump(weather_data, handle)


if __name__ == '__main__':
    example()
