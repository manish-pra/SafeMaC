# Near-Optimal Multi-Agent Learning for Safe Coverage Control

The repository contains all code and experiments for MacOpt and SafeMac.


## Dependencies
1. The code is tested on Python 3.8.5 and 3.8.10
2. On the cluster, you can load  gcc/8.2.0 python/3.8.5 ffmpeg/5.0
3. Install packages from requirements.txt
4. For plotting clone the repo [plotting_utilities](https://github.com/befelix/plotting_utilities) in apps folder 

## To Run
1. Set the following params in main.py; by default, it is set to run gorilla environment; change it to 'GP' or 'obstacles' as per your wish
```  
workspace = "SafeMaC"
env_load_path = workspace + \
    "/experiments/gorilla/environments/env_" + \
    str(args.env) + "/"
```
2. python3 SafeMaC/main.py -i $i -env $env_i -param $param_i
```
$param_i = Name of the param file (see params folder) to pick an algorithm and the environment type
$env_i = an integer to pick an instance of the environment
$i = an integer to run multiple instances
```

3. Following example runs MacOpt on the gorilla environment with the 1st environment instance

 ``` 
 python3 SafeMaC/main.py -i 1 -env 1 -param "smcc_MacOpt_gorilla"
```
4. Run the following commands from your workspace to use the plotting scripts just outside of SafeMaC
```
python3 SafeMaC/apps/bar_chart.py
python3 SafeMaC/apps/plotting_script.py
```
5. Each env folder contains an image of the environment, and on running the experiment, it will produce a .mp4 along with some plots in the experiment folder
1. Further, you can use the consolidate_data.py script and the plotting scripts in the apps folder to plot the results. Currently, their path is set to the pre-trained data folder.
1. In the plotting script, comment the lines between 7 to 31, as per the environment and algorithm you want to plot

## Visualizations
1. Running the code will generate visuals in .mp4 format. A few of them are in the Visualizations folder for readers' reference. Due to upload size limitations, they are not currently present in the experiment folders but only in the visualization folder
1. For environement plots check environement.py script
## Repository structure
    .
    ├── apps
    │   ├── consolidate_data.py          # Consolidate all the data produced in different folders to a single folder
    │   ├── plotting_script.py
    │   ├── bar_chart.py
    ├── experimenets                   # Each game have a saparate folder with this structure
    │   ├── Obstacles                     
    │   ├── Gorilla                
    │   ├── GP
    ├── utils
    │   ├── agent_helper.py 
    │   ├── agent.py 
    │   ├── central_graph.py 
    │   ├── common.py 
    │   ├── datatypes.py 
    │   ├── environement.py 
    │   ├── ground_truth.py 
    │   ├── helper.py 
    │   ├── initializer.py 
    │   ├── visu.py 
    ├── params
    │   ├── smcc_MacOpt_gorilla.yaml 
    │   ├── smcc_UCB_gorilla.yaml 
    │   ├── smcc_MacOpt_GP.yaml 
    │   ├── smcc_UCB_GP.yaml  
    │   ├── smcc_SafeMac_GP.yaml 
    │   ├── smcc_PassiveMac_GP.yaml  
    │   ├── smcc_TwoStage_GP.yaml  
    │   ├── smcc_SafeMac_gorilla.yaml 
    │   ├── smcc_PassiveMac_gorilla.yaml  
    │   ├── smcc_TwoStage_gorilla.yaml  
    │   ├── smcc_SafeMac_obstacles.yaml 
    │   ├── smcc_PassiveMac_obstacles.yaml  
    │   ├── smcc_TwoStage_obstacles.yaml  
    ├── main.py
    └── ...

