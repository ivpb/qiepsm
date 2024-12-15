import os
import torch
import random
import numpy as np
from utils.datasets import DatasetLoader
from utils.utils import DATASET_ROOT_FOLDER, DEVICE_IDS, get_train_val_fraction

PRETRAINED_MODEL_NAME = "bert-base-uncased"
FINE_TUNE_LAYERS = 4

trec19_loader = DatasetLoader(
    task="evaluate",
    pretrained_model_name=PRETRAINED_MODEL_NAME,
    corpus_path=f"{DATASET_ROOT_FOLDER}/MSMARCO/msmarco.trec2019dl.corpus.json",
    query_path=f"{DATASET_ROOT_FOLDER}/MSMARCO/msmarco.trec2019dl.json",
    splits=[],
    batch_size=1,
    seed=42
)

trec20_loader = DatasetLoader(
    task="evaluate",
    pretrained_model_name=PRETRAINED_MODEL_NAME,
    corpus_path=f"{DATASET_ROOT_FOLDER}/MSMARCO/msmarco.trec2020dl.corpus.json",
    query_path=f"{DATASET_ROOT_FOLDER}/MSMARCO/msmarco.trec2020dl.json",
    splits=[],
    batch_size=1,
    seed=42
)

def run_evaluation(execute, run, train_val_fraction):
    SEED = 42+(run-1)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    train_dataset_loader = DatasetLoader(
        task="best_fit",
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        corpus_path=f"{DATASET_ROOT_FOLDER}/MSMARCO/msmarco.sbert.train.full.corpus.json",
        query_path=f"{DATASET_ROOT_FOLDER}/MSMARCO/msmarco.sbert.train.full.json",
        splits=[100000, 10000],
        batch_size=4*16,
        seed=SEED,
        train_val_fraction = train_val_fraction
    )

    directory_path = f"./runs/{int(train_val_fraction*100)}/{run}"
    os.makedirs(directory_path, exist_ok=True)
    execute(directory_path,train_dataset_loader)
