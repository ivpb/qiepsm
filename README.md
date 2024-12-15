# Quantum-Inspired Embeddings Projection and Similarity Metrics

This repository provides the code for the publication paper [link](https://link.com). The structure of the repository is organized as follows:

- The directory [compression experiments](https://github.com/ivpb/qiepsm/tree/main/compression%20experiments) contains code for experiments performed using a fixed dataset size.
- The directory [training experiments](https://github.com/ivpb/qiepsm/tree/main/training%20experiments) contains code for experiments conducted using a variable dataset size.
- Subfolders with the suffixes `_base` and `_v5` contain the code for models built on the backbone models `bert-base-uncased` and `msmarco-distilbert-cos-v5`, respectively.
- Each subfolder is further divided into subdirectories for quantum-inspired and classical approaches. Quantum-inspired models are located in folders prefixed with `qi`.

## Dependencies

All required libraries are listed in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```

## Running Experiments

Before running experiments, ensure that you download the required [datasets](https://github.com/ivpb/qiepsm#datasets). Each experiment directory contains a `runner.py` file, which can be executed using:

```bash
python runner.py
```

The results are stored in the `runs` directory as [log files](https://github.com/ivpb/qiepsm#log-files). By default, the `runner.py` script executes experiments sequentially, but you can modify it for parallel execution to reduce runtime. You can also navigate to the specific model folder (e.g., `cd "compression experiments/compression_base/bert base compressed"`) and run the corresponding Python script directly, such as:

```bash
python bert.py
```

## Log Files

After each training epoch, the model is evaluated on the benchmark datasets TREC 2019 DL and TREC 2020 DL using the NDCG@10 metric. Training and evaluation logs are saved in a `.txt` file. The format of the log file is as follows:

```
Start Training: [date time]
Model: [name], Pretrained Model: [name], Fine Tune Layers: [count]
Splits: [train., val.], Batch: [size], GPU: [IDs]
Epoch _, Train Loss: [training loss], Train Accuracy: [training accuracy]
Epoch _, Val Loss: [validation loss], Val Accuracy: [validation accuracy]
Epoch _, Evaluate: TREC19, NDCG@10: _
Epoch _, Evaluate: TREC20, NDCG@10: _
End Training: [date time]
```

Key abbreviations include:
- `Train.`: Training dataset size
- `Val.`: Validation dataset size
- `GPU`: CUDA device IDs used during training

Any information outside this format in the log files can be ignored as irrelevant.

## Datasets

Download the "Datasets" folder from this link: [Download Datasets](https://seafile.rlp.net/d/0ef4382eff3a42b3a2fd/), and place it in a directory of your choice. Copy the absolute path to this folder and set it in each `utils/utils.py` file by assigning it to the constant variable `DATASET_ROOT_FOLDER`. For example:

```python
DATASET_ROOT_FOLDER = "/path to datasets/datasets"
```

The folder contains the training datasets as well as benchmark datasets used for evaluation.