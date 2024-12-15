from transformers import AutoModel
import torch.nn.functional as F
from torch import tensor
import torch.nn as nn
import torch

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.trainer import BertTrainer
from utils.evaluator import BertEvaluator
from utils.utils import fidelity, DEVICE
from compression_v5.common import DEVICE_IDS, run_evaluation, PRETRAINED_MODEL_NAME, FINE_TUNE_LAYERS, trec19_loader, trec20_loader


class ZYZ(nn.Module):
    def __init__(self,n_qubits):
        super(ZYZ, self).__init__()
        self.alpha = nn.Parameter(torch.randn(n_qubits),requires_grad=True)
        self.beta = nn.Parameter(torch.randn(n_qubits),requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(n_qubits),requires_grad=True)
        self.delta = nn.Parameter(torch.randn(n_qubits),requires_grad=True)
        self.identity = self.u_composition(torch.zeros(n_qubits),torch.zeros(n_qubits),torch.zeros(n_qubits),torch.zeros(n_qubits))

    def u_composition(self,alpha,beta,gamma,delta):
        cos_gamma = torch.cos(gamma/2)
        sin_gamma = torch.sin(gamma/2)
        e_0_0 = torch.exp(1j*(alpha - beta/2 - delta/2)) * cos_gamma
        e_0_1 = -torch.exp(1j*(alpha - beta/2 + delta/2)) * sin_gamma
        e_1_0 = torch.exp(1j*(alpha + beta/2 - delta/2)) * sin_gamma
        e_1_1 = torch.exp(1j*(alpha + beta/2 + delta/2)) * cos_gamma
        return torch.stack((torch.cat((e_0_0.unsqueeze(dim=-1), e_0_1.unsqueeze(dim=-1)),dim=-1),torch.cat((e_1_0.unsqueeze(dim=-1), e_1_1.unsqueeze(dim=-1)),dim=-1)),dim=-2)

    def tanh(self,t,alpha):
        return torch.tanh(t)*alpha/2 + torch.ones_like(t)*alpha/2

    def forward(self,qubits,controlled=False):
        alpha = self.tanh(self.alpha,4*torch.pi)
        beta = self.tanh(self.beta,4*torch.pi)
        gamma = self.tanh(self.gamma,4*torch.pi)
        delta = self.tanh(self.delta,4*torch.pi)
        U = self.u_composition(alpha,beta,gamma,delta)
        if controlled:
            CU = F.pad(U, (2,0,2,0), "constant", 0.0) + F.pad(self.identity, (0,2,0,2), "constant", 0.0).to(DEVICE)
            return CU.matmul(qubits.unsqueeze(-1)).squeeze(-1)
        return U.matmul(qubits.unsqueeze(-1)).squeeze(-1)


class QiBertEmbedder(nn.Module):
    def __init__(self, pretrained_model_name):
        super(QiBertEmbedder, self).__init__()
        self.pretrained_model_name = pretrained_model_name
        self.bert = AutoModel.from_pretrained(pretrained_model_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.n_qubits = 256
        self.a_zyz = ZYZ(self.n_qubits)
        self.b_zyz = ZYZ(self.n_qubits)
        self.c_zyz = ZYZ(self.n_qubits)
        self.b_cb_zyz = ZYZ(self.n_qubits)

        self.cb_c_zyz = ZYZ(self.n_qubits)
        self.ba_c_zyz = ZYZ(self.n_qubits)

    def mean_pool(self,embeddings, attention_masks):
        in_mask = attention_masks.unsqueeze(-1).expand(
            embeddings.size()
        ).float()
        pool = torch.sum(embeddings * in_mask, 1) / torch.clamp(
            in_mask.sum(1), min=1e-9
        )
        return pool

    def embedder_mag(self, batch):
        input_ids = batch[:,0]
        attention_masks = batch[:,1]
        embeddings = torch.stack([self.mean_pool(self.bert(input_ids[:,i,:], attention_mask=attention_masks[:,i,:]).last_hidden_state,attention_masks[:,i,:]) for i in range(input_ids.shape[1])],dim=1)
        return F.normalize(embeddings,dim=2)

    def compute_qstate(self,bert_batch):
        ae = torch.tanh(self.embedder_mag(bert_batch))
        be = torch.zeros_like(ae)
        embedding_mags = ae*torch.pi/2 + torch.ones_like(ae)*torch.pi/2
        embedding_phases = be*(2*torch.pi-tensor(10).pow(-9.0))/2 + torch.ones_like(be)*(2*torch.pi-tensor(10).pow(-9.0))/2

        a = torch.unsqueeze(torch.cos(embedding_mags/2),dim=3)
        b = torch.unsqueeze(torch.sin(embedding_mags/2) * torch.exp(1j*embedding_phases),dim=3)
        qubits = torch.cat((a,b),dim=3)
        return qubits

    def forward(self, batch):
        qubits = self.compute_qstate(batch)
        a = qubits[:,:,:self.n_qubits,:]
        b = qubits[:,:,self.n_qubits:self.n_qubits*2,:]
        c = qubits[:,:,self.n_qubits*2:,:]

        # zyz
        cb_qubits = torch.einsum('bnki,bnkj->bnkij', self.c_zyz(c), self.b_zyz(b)).view(b.shape[0],b.shape[1],b.shape[2],b.shape[3]*2)
        # CUnitary
        cb_qubits = self.cb_c_zyz(cb_qubits,controlled=True)

        # read b
        b_cb = (cb_qubits[:,:,:,0:2].conj()*cb_qubits[:,:,:,0:2] + cb_qubits[:,:,:,2:].conj()*cb_qubits[:,:,:,2:]).sqrt()
        # zyz
        ba_qubits = torch.einsum('bnki,bnkj->bnkij', self.b_cb_zyz(b_cb), self.a_zyz(a)).view(a.shape[0],a.shape[1],a.shape[2],a.shape[3]*2)
        # CUnitary
        ba_qubits = self.ba_c_zyz(ba_qubits,controlled=True)

        # read a
        a_ba = (ba_qubits[:,:,:,0:2].conj()*ba_qubits[:,:,:,0:2] + ba_qubits[:,:,:,2:].conj()*ba_qubits[:,:,:,2:]).sqrt()

        return torch.view_as_real(a_ba)

    def get_description(self):
        return f"Model: qi-bert (scaled), Pretrained Model: {self.pretrained_model_name}, Qi Head: U+CU"


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
    trec19_res = trec19_evaluator.evaluate().tolist()
    trec20_res = trec20_evaluator.evaluate().tolist()
    return [log("TREC19",trec19_res), log("TREC20",trec20_res)],float('%.4f'%((trec19_res[0]+trec20_res[0])/2))

def execute(director_path,train_dataset_loader):
    trainer = BertTrainer(
        task="best_fit",
        model=QiBertEmbedder(PRETRAINED_MODEL_NAME),
        dataset_loader=train_dataset_loader,
        score_fn=score_fn,
        epoch=5,
        model_path=None,#f"{director_path}/../qi-bert-{director_path.split('/')[-1]}-{{test_acc}}#.pth",
        log_path=f"{director_path}/qi-bert.txt",
        after_each_epoch=after_each_epoch,
        device_ids=DEVICE_IDS
    )

    trainer.train()

def start_run(run):
    run_evaluation(execute,run)