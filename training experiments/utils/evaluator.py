import torch
from utils.trainer import BertTrainer
from utils.datasets import DatasetLoader
from utils.utils import DEVICE, compute_ndcg

class BertEvaluator:

    def __init__(self, model, dataset_loader: DatasetLoader, score_fn,device_ids=[]):
        self.model = model
        self.dataset_loader = dataset_loader
        self.score_fn = score_fn
        self.device_ids = device_ids

    def evaluate(self):
        if isinstance(self.model,torch.nn.DataParallel):
            model = self.model
        else:
            model = torch.nn.DataParallel(self.model,device_ids=self.device_ids)
            model.to(DEVICE)

        test_loader,_ = self.dataset_loader.get_loaders()
        return self._evaluate(model,test_loader)

    def _evaluate(self, model, test_loader):
        model.eval()
        k = 0
        total_scores = torch.tensor([0.0,0.0,0.0]).to(DEVICE)
        with torch.no_grad():
            for batch,label_batch in test_loader:
                label_batch = label_batch.to(DEVICE)
                labels = label_batch.squeeze(dim=0)
                if self.dataset_loader.with_idf:
                    batch_size = batch[0].shape[0]
                    batch = torch.stack((batch[0][:,0].squeeze(dim=0).unsqueeze(dim=1), batch[0][:,1].squeeze(dim=0).unsqueeze(dim=1)),dim=1),batch[1].squeeze(dim=0).unsqueeze(dim=1)
                else:
                    batch_size = batch.shape[0]
                    batch = torch.stack((batch[:,0].squeeze(dim=0).unsqueeze(dim=1), batch[:,1].squeeze(dim=0).unsqueeze(dim=1)),dim=1)
                assert batch_size==1, "Batch size of evaluation dataset loader should be one"
                k += batch_size
                chunk_into_minibatch = lambda batch: [torch.cat((batch[0].unsqueeze(dim=0),minibatch),dim=0) for minibatch in torch.split(batch[1:],1024)]
                process_minibatch = lambda minibatch: BertTrainer.process_batch(model,minibatch,self.score_fn,self.dataset_loader.with_idf).squeeze(dim=1)
                minibatches = zip(chunk_into_minibatch(batch[0]),chunk_into_minibatch(batch[1])) if self.dataset_loader.with_idf else chunk_into_minibatch(batch)
                scores = torch.cat([process_minibatch(minibatch) for minibatch in minibatches],dim=-1)                

                total_scores += torch.tensor([compute_ndcg(scores,labels,10)]).to(DEVICE)
        return total_scores/k
