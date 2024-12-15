import argparse
import os
import importlib.util
import sys
from itertools import product
from importlib import import_module

import utils.utils

model_to_dirs = {

    "base_bec": "compression_base/bert base compressed",
    "base_qbec": "compression_base/qi bert base compressed",

}

def execute_run(model, run, train_val_fraction):
    last_dir = os.getcwd()
    os.chdir(model_to_dirs[model])

    filename = [f for f in os.listdir() if f.endswith(".py")][0]
    file_path = os.path.join("./",filename)
    
    spec = importlib.util.spec_from_file_location(filename, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    start_run = getattr(module,"start_run")
    start_run(run, train_val_fraction)

    os.chdir(last_dir)


train_val_fractions = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
train_val_fractions.reverse()
for model in ["base_bec","base_qbec"]:
    for run_idx in range(1,21):
        for train_val_fraction in train_val_fractions:
            execute_run(model, run_idx, train_val_fraction)
