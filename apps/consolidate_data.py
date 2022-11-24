import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

workspace = "SafeMaC"
path = workspace + "/experiments/gorillaLA3D50-e/environments"

env_folders = [f.name for f in os.scandir(path) if f.is_dir()]

all_data = {}
all_env_data = {}
for env in env_folders:
    all_env_data[env] = {}
    path_env_i = path + "/" + env
    class_folders = [f.name for f in os.scandir(path_env_i) if f.is_dir()]
    for class_ in class_folders:
        all_env_data[env][class_] = {}
        path_class = path_env_i + "/" + class_
        experiments_folders = [
            f.name for f in os.scandir(path_class) if f.is_dir()]
        for exp_num in experiments_folders:
            data_path = path_class + "/" + exp_num + "/data.pkl"
            if os.path.exists(data_path):
                if os.path.getsize(data_path) != 0:
                    k = open(data_path, "rb")
                    data = pickle.load(k)
                    k.close()
                    all_env_data[env][class_][exp_num] = data
        print(env, " ", class_, " ", len(experiments_folders), "done")

a_file = open(path + "/50_env_data.pkl", "wb")
pickle.dump(all_env_data, a_file)
a_file.close()
