from transformers import AutoModel
from torch import tensor
import torch.nn as nn
import torch

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.trainer import BertTrainer
from utils.evaluator import BertEvaluator
from utils.utils import fidelity
from bert_base.common import DEVICE_IDS, run_evaluation, PRETRAINED_MODEL_NAME, FINE_TUNE_LAYERS, trec19_loader, trec20_loader


class QiBertEmbedder(nn.Module):
    def __init__(self, pretrained_model_name,fine_tune_layers):
        super(QiBertEmbedder, self).__init__()
        self.fine_tune_layers = fine_tune_layers
        self.pretrained_model_name = pretrained_model_name
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        for i in range(fine_tune_layers):
            for param in self.bert.encoder.layer[-i].parameters():
                param.requires_grad = True

    def mean_pool(self,embeddings, attention_masks):
        in_mask = attention_masks.unsqueeze(-1).expand(
            embeddings.size()
        ).float()
        pool = torch.sum(embeddings * in_mask, 1) / torch.clamp(
            in_mask.sum(1), min=1e-9
        )
        return pool

    def forward(self, batch):
        input_ids = batch[:,0]
        attention_masks = batch[:,1]
        embeddings = torch.stack([self.mean_pool(self.bert(input_ids[:,i,:], attention_mask=attention_masks[:,i,:]).last_hidden_state,attention_masks[:,i,:]) for i in range(input_ids.shape[1])],dim=1)

        ae = torch.tanh(embeddings)
        be = torch.zeros_like(ae)
        embedding_mags = ae*torch.pi/2 + torch.ones_like(ae)*torch.pi/2
        embedding_phases = be*(2*torch.pi-tensor(10).pow(-9.0))/2 + torch.ones_like(be)*(2*torch.pi-tensor(10).pow(-9.0))/2

        a = torch.unsqueeze(torch.cos(embedding_mags/2),dim=3)
        b = torch.unsqueeze(torch.sin(embedding_mags/2) * torch.exp(1j*embedding_phases),dim=3)
        qubits = torch.cat((a,b),dim=3)
        return torch.view_as_real(qubits)

    def get_description(self):
        return f"Model: qi-bert (scaled), Pretrained Model: {self.pretrained_model_name}, Fine Tune Layers: {self.fine_tune_layers}"


def score_fn(model,question_bert_batch, passage_bert_batch):
    a = torch.view_as_complex(model(question_bert_batch))
    b = torch.view_as_complex(model(passage_bert_batch))
    return fidelity(a,b)*20.0

def evaluator_score_fn(model,question_passage_bert_batch, _):
    output = torch.view_as_complex(model(question_passage_bert_batch))
    return fidelity(output[:1],output[1:])*20.0

def after_each_epoch(model,epoch,max_epoch):
    log = lambda name,res: f"Epoch {epoch}/{max_epoch}, Evaluate: {name}, NDCG@10: {res[0]:.4f}"
    trec19_evaluator = BertEvaluator(
        model=model,
        dataset_loader=trec19_loader,
        score_fn=evaluator_score_fn,
        device_ids=DEVICE_IDS
    )
    trec20_evaluator = BertEvaluator(
        model=model,
        dataset_loader=trec20_loader,
        score_fn=evaluator_score_fn,
        device_ids=DEVICE_IDS
    )
    return [log("TREC19",trec19_evaluator.evaluate().tolist()),log("TREC20",trec20_evaluator.evaluate().tolist())]

def execute(director_path,train_dataset_loader):
    trainer = BertTrainer(
        task="best_fit",
        model=QiBertEmbedder(PRETRAINED_MODEL_NAME,FINE_TUNE_LAYERS),
        dataset_loader=train_dataset_loader,
        score_fn=score_fn,
        epoch=5,
        model_path=None,
        #model_path=f"{director_path}/qi-bert-{{epoch}}.pth",
        log_path=f"{director_path}/qi-bert.txt",
        after_each_epoch=after_each_epoch,
        device_ids=DEVICE_IDS,
        lr_optimized=True
    )

    trainer.train()

def start_run(run):
    run_evaluation(execute,run)
