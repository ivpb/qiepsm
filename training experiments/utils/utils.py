import torch
from torchmetrics.functional.retrieval import retrieval_normalized_dcg

DATASET_ROOT_FOLDER = '/path to datasets/datasets'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:",DEVICE)

__train_val_fraction = 1.0

def get_train_val_fraction():
    return __train_val_fraction

def set_train_val_fraction(fraction):
    __train_val_fraction = fraction


def get_available_gpu_ids():
    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus))
    return gpu_ids

DEVICE_IDS = get_available_gpu_ids()



def compute_ndcg(preds,targets,top_k):
    return retrieval_normalized_dcg(preds, targets,top_k=top_k)

def fidelity(b_q_qubits,b_ps_qubits):
    b_q_qubits = b_q_qubits.conj()
    return ((b_q_qubits[:,:,:,0] * b_ps_qubits[:,:,:,0]) + (b_q_qubits[:,:,:,1] * b_ps_qubits[:,:,:,1])).prod(dim=2).abs().square()

torch_cos_sim = torch.nn.CosineSimilarity(dim=2)
def cos_sim(b_q_embs,b_ps_embs):
    return torch_cos_sim(b_q_embs,b_ps_embs)