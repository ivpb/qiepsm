import glob
import os
import math
from datetime import datetime

import torch
import torch.optim as optim
from tqdm.auto import tqdm

from utils.datasets import DatasetLoader
from utils.utils import DEVICE, compute_ndcg



class BertTrainer:

    def __init__(self, task, model, dataset_loader: DatasetLoader, score_fn, epoch, model_path, log_path, device_ids=[], after_each_epoch=None, continue_at_epoch=None,lr_optimized=False):
        self.task = task
        self.model = model
        self.dataset_loader = dataset_loader
        self.epoch = epoch
        self.score_fn = score_fn
        self.log_path = log_path
        self.model_path = model_path
        self.device_ids = device_ids
        self.after_each_epoch = after_each_epoch
        self.continue_at_epoch = continue_at_epoch
        self.optimization_step = math.ceil(self.dataset_loader.batch_size/self.dataset_loader.mini_batch_size) if self.dataset_loader.mini_batch_size else 1
        self.lr_optimized = lr_optimized

    def train(self):
        self._log("Start Training: "+datetime.now().astimezone().isoformat())
        self._log(self.model.get_description())
        self._log(f"Dataset: {self.dataset_loader.get_dataset_name()}, Splits: {self.dataset_loader.splits}, Batch: {self.dataset_loader.batch_size}, GPU: {self.device_ids}, Passage: {self.dataset_loader.passage_count}, IDF: {self.dataset_loader.with_idf}, Step: {self.optimization_step}")

        model = self.model
        train_loader,dev_loader = self.dataset_loader.get_loaders()

        prev_epoch = 0
        if self.continue_at_epoch is not None and self.model_path:
            model,prev_epoch = self.load_state_dict(self.model_path.replace("{epoch}",str(self.continue_at_epoch)),model)

        model = torch.nn.DataParallel(model,device_ids=self.device_ids)

        model.to(DEVICE)
        if self.lr_optimized:
            optimizer = optim.Adam(model.parameters(), lr=4e-5)
            print("!!!lr optimized!!!")
        else:
            optimizer = optim.Adam(model.parameters(), lr=2e-5)
            print("!!!no lr optimized!!!")

        criterion = torch.nn.CrossEntropyLoss() if self.task=="best_fit" else None
        for epoch in range(prev_epoch+1,self.epoch+1):
            train_loss,train_accuracy = self._train(model, train_loader, optimizer, criterion, epoch)
            self._log(f"Epoch {epoch}/{self.epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            if self.model_path and "{test_acc}" in self.model_path:
                existing_files = glob.glob(self.model_path.replace("{test_acc}","*"))
                assert len(existing_files)<=1, f"Ambiguous models: {existing_files}"

                dev_loss,dev_accuracy = self._validate(model, dev_loader, criterion)
                self._log(f"Epoch {epoch}/{self.epoch}, Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}")
                msgs,test_accuracy = self.after_each_epoch(model,epoch,self.epoch)
                for msg in msgs:
                    self._log(msg)
                if len(existing_files)==0 or float(existing_files[0].split("-")[-1].split("#")[0]) < test_accuracy:
                    if len(existing_files)>=1:
                        os.remove(existing_files[0])
                    self.save_state_dict(self.model_path.replace("{test_acc}",str(test_accuracy)),model,epoch)
            else:
                if self.model_path:
                    self.save_state_dict(self.model_path.replace("{epoch}",str(epoch)),model,epoch)
                dev_loss,dev_accuracy = self._validate(model, dev_loader, criterion)
                self._log(f"Epoch {epoch}/{self.epoch}, Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}")
                if self.after_each_epoch:
                    msgs = self.after_each_epoch(model,epoch,self.epoch) or []
                    for msg in (msgs[0] if len(msgs)>=2 and (isinstance(msgs[0],list) or isinstance(msgs[0],tuple)) else msgs):
                        self._log(msg)

        self._log("End Training: "+datetime.now().astimezone().isoformat())
        return model

    def _log(self,msg):
        log_file = open(self.log_path,"a")
        log_file.write(msg+"\n")
        print(msg)
        log_file.close()

    def _train(self, model, train_loader, optimizer, criterion, epoch):
        model.train()
        k = 0
        step = 0
        total_loss = 0.0
        total_accuracy = 0.0
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            step += 1
            label_batch = batch[1] if self.task=="rank" else None
            batch = batch[0] if self.task=="rank" else batch
            batch_size = (batch[0] if self.dataset_loader.with_idf else batch).shape[0]
            k += batch_size
            scores = self.process_batch(model,batch,self.score_fn,self.dataset_loader.with_idf)

            if self.task=="best_fit":
                labels = torch.zeros(scores.shape[0], dtype=torch.long).to(DEVICE)
            else:
                labels = label_batch.to(DEVICE)

            loss = criterion(scores, labels)/self.optimization_step
            loss.backward()
            if step%self.optimization_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if self.task=="best_fit":
                accuracy = scores.argmax(dim=1).eq(labels).float().mean()
            else:
                accuracy = compute_ndcg(scores,labels,top_k=self.dataset_loader.passage_count)
            total_accuracy += accuracy.item() * batch_size
            total_loss += loss.item()

            loop.set_description(f'Epoch {epoch}/{self.epoch}')
            loop.set_postfix(loss=loss.item(),acc=accuracy.item())
        if step > 0 and step%self.optimization_step != 0:
            optimizer.step()
            optimizer.zero_grad()
        return total_loss / k, total_accuracy / k

    @staticmethod
    def process_batch(model,batch,score_fn,with_idf):
        bert_batch = batch[0] if with_idf else batch
        bert_batch.to(DEVICE)

        bert_batch_ids, bert_batch_masks = bert_batch[:,0], bert_batch[:,1]
        bert_batch_ids = bert_batch_ids.squeeze(dim=2)
        bert_batch_masks = bert_batch_masks.squeeze(dim=2)

        question_bert_batch = torch.stack((bert_batch_ids[:,0:1], bert_batch_masks[:,0:1]),dim=1)
        passage_bert_batch = torch.stack((bert_batch_ids[:,1:], bert_batch_masks[:,1:]),dim=1)

        if with_idf:
            idf_batch = batch[1]
            idf_batch.to(DEVICE)
            question_idf_batch = idf_batch[:,0:1]
            passage_idf_batch = idf_batch[:,1:]
            return score_fn(model,question_bert_batch, passage_bert_batch, question_idf_batch, passage_idf_batch)
        else:
            return score_fn(model, question_bert_batch, passage_bert_batch)

    def _validate(self, model, dev_loader, criterion):
        model.eval()
        with torch.no_grad():
            k = 0
            total_loss = 0.0
            total_accuracy = 0.0
            for batch in dev_loader:
                label_batch = batch[1] if self.task=="rank" else None
                batch = batch[0] if self.task=="rank" else batch
                batch_size = (batch[0] if self.dataset_loader.with_idf else batch).shape[0]
                k += batch_size
                scores = self.process_batch(model,batch,self.score_fn,self.dataset_loader.with_idf)

                if self.task=="best_fit":
                    labels = torch.zeros(scores.shape[0], dtype=torch.long).to(DEVICE)
                else:
                    labels = label_batch.to(DEVICE)

                loss = criterion(scores, labels)
                if self.task=="best_fit":
                    accuracy = scores.argmax(dim=1).eq(labels).float().mean()
                else:
                    accuracy = compute_ndcg(scores,labels,top_k=self.dataset_loader.passage_count)
                total_accuracy += accuracy.item() * batch_size
                total_loss += loss.item()

        return total_loss / k, total_accuracy / k

    @staticmethod
    def save_state_dict(model_path,model,epoch):
        torch.save({'epoch': epoch,
                    'model_state_dict': (model.module if hasattr(model, 'module') else model).state_dict()},
                   model_path)

    @staticmethod
    def load_state_dict(model_path,model):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        return model,epoch
