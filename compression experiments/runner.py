import os
import importlib.util
from importlib import import_module

model_to_dirs = {
    "base_bt": "bert_base/bert base trained",
    "base_qbt": "bert_base/qi bert base trained",
    "v5_bt": "bert_v5/bert trained",
    "v5_qbt": "bert_v5/qi bert trained",

    "base_bec": "compression_base/bert base compressed",
    "base_qbec": "compression_base/qi bert base compressed",
    "v5_bec": "compression_v5/bert compressed",
    "v5_qbec": "compression_v5/qi bert compressed",
    "v5_bfec": "compression_v5/bert frozen compressed",
    "v5_qbfec": "compression_v5/qi bert frozen compressed",


    # News models

    "base_becr_64": "compression_v5/bert base compressed 64",
    "base_qbecr_64": "compression_v5/qi bert base compressed 64",
    "base_becr_128": "compression_v5/bert base compressed 128",
    "base_qbecr_128": "compression_v5/qi bert base compressed 128",
    "base_becr_384": "compression_v5/bert base compressed 384",
    "base_qbecr_384": "compression_v5/qi bert base compressed 384",

    "v5_becr_64": "compression_v5/bert compressed 64",
    "v5_qbecr_64": "compression_v5/qi bert compressed 64",
    "v5_becr_128": "compression_v5/bert compressed 128",
    "v5_qbecr_128": "compression_v5/qi bert compressed 128",
    "v5_becr_384": "compression_v5/bert compressed 384",
    "v5_qbecr_384": "compression_v5/qi bert compressed 384",

}

def execute_run(model,run):
    last_dir = os.getcwd()
    os.chdir(model_to_dirs[model])

    filename = [f for f in os.listdir() if f.endswith(".py")][0]
    file_path = os.path.join("./",filename)
    
    spec = importlib.util.spec_from_file_location(filename, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    start_run = getattr(module,"start_run")
    start_run(run)

    os.chdir(last_dir)


for model in model_to_dirs.keys():
    for run_idx in range(1,21):
        execute_run(model,run_idx)