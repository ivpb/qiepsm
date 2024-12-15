import os
import torch
import random
import numpy as np
from utils.datasets import DatasetLoader
from utils.utils import DATASET_ROOT_FOLDER, DEVICE_IDS

PRETRAINED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-cos-v5"
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

def run_evaluation(execute,run):
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
        seed=SEED
    )

    directory_path = f"./runs/{run}"
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    execute(directory_path,train_dataset_loader)
