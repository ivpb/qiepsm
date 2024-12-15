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
from compression_base.common import DEVICE_IDS, run_evaluation, PRETRAINED_MODEL_NAME, trec19_loader, trec20_loader


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

        self.dims = [[[0,384],[384,768]],[0,384]]
        tstr = lambda dims: "_".join([str(d) for d in dims])
        for i,(dims1,dims2) in enumerate(self.dims[:-1][::-1]):
            i = len(self.dims[:-1]) - i - 1
            setattr(self,"u_"+tstr(dims1)+"_"+str(i),ZYZ(dims1[1]-dims1[0]))
        for i,(dims1,dims2) in enumerate(self.dims[:-1]):  
            setattr(self,"u_"+tstr(dims2)+"_"+str(i),ZYZ(dims2[1]-dims2[0]))
        for i,(dims1,dims2) in enumerate(self.dims[:-1]):
            setattr(self,"cu_"+tstr(dims1)+"_"+tstr(dims2)+"_"+str(i),ZYZ(dims1[1]-dims1[0]))

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
        return embeddings#F.normalize(embeddings,dim=2)

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
        tstr = lambda dims: "_".join([str(d) for d in dims])
        for i,(dims1,dims2) in enumerate(self.dims[:-1]):
            a,b = qubits[:,:,dims1[0]:dims1[1],:],qubits[:,:,dims2[0]:dims2[1],:]
            u_a = getattr(self,"u_"+tstr(dims1)+"_"+str(i))
            u_b = getattr(self,"u_"+tstr(dims2)+"_"+str(i))
            cu_ab = getattr(self,"cu_"+tstr(dims1)+"_"+tstr(dims2)+"_"+str(i))

            # zyz
            ba_qubits = torch.einsum('bnki,bnkj->bnkij', u_b(b), u_a(a)).view(a.shape[0],a.shape[1],a.shape[2],a.shape[3]*2)
            # CUnitary
            ba_qubits = cu_ab(ba_qubits,controlled=True)

            # read a
            qubits[:,:,dims1[0]:dims1[1],:] = (ba_qubits[:,:,:,0:2].conj()*ba_qubits[:,:,:,0:2] + ba_qubits[:,:,:,2:].conj()*ba_qubits[:,:,:,2:]).sqrt()

        return torch.view_as_real(qubits[:,:,self.dims[-1][0]:self.dims[-1][1],:])

    def get_description(self):
        return f"Model: qi-bert (adj), Pretrained Model: {self.pretrained_model_name}, Fine Tune Layers: {self.fine_tune_layers}, Qi Head: U+CU(384)"


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
        model=QiBertEmbedder(PRETRAINED_MODEL_NAME,4),
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
