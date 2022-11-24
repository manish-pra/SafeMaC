Code for the environment of SafeMAC. Weather data is from [Open Weather Map](https://openweathermap.org).
Density data come from the fitted model in [this paper](https://arxiv.org/pdf/2110.11181.pdf).
The original gorilla data set comes from [here](https://www.rdocumentation.org/packages/spatstat/versions/1.52-1/topics/gorillas). 

**NOTE**: To fetch data from Open Weather Map, you should store in APIKey.txt your OWM API key.

### Overview:
 - geodeisc_grid.py: code to generate grid of coordinates.
 - owm.py: code to make calls to Open Weather Map API to get weather data
 - fetch_data.py: code that uses scripts above to create grid of coordinates from
 specific locations and queries and saves the corresponding weather data. It also 
 contains utilities for loading such data.
  - KGS_environment.py: Code implementing basic functionality for grid_functions.
  Once we load a base function (e.g. raw weather data or fitted gorilla model),
  we can upsample, downsample, or crop it to fit to a desired shape.
  - .json files: contain coordinates of locations of interest in decimal format.
  
### Basic usage
 - Fetch and save data using fetch_data.py by specifying the lower left corner of the 
coordinate grid, the width of the step in x and y and the number of steps.
 - Use the example in KGS_environment.py to create grid_function interfaces for
 density and constraint.

