import random

import torch
import json
import functools
from torch import tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

class DatasetLoader:

    def __init__(self,task,pretrained_model_name,corpus_path,query_path,splits,batch_size,seed,hard_neg_count=None,passage_count=None,with_idf=False,idf_max_features=5000,idf_pca_components=1000,mini_batch_size=None,train_val_fraction=1):
        self.task = task
        self.seed = seed
        if splits and train_val_fraction:
            self.splits = [int(splits[0] * train_val_fraction), int(splits[1] * train_val_fraction)]
            print(f"Using train_val_fraction {train_val_fraction}")
        else:
            self.splits = splits
        self.with_idf = with_idf
        self.batch_size = batch_size
        self.query_path = query_path
        self.corpus_path = corpus_path
        self.hard_neg_count = hard_neg_count
        self.passage_count = passage_count
        self.idf_max_features = idf_max_features
        self.idf_pca_components = idf_pca_components
        self.pretrained_model_name = pretrained_model_name
        self.mini_batch_size = mini_batch_size if mini_batch_size else self.batch_size

        self.generator = torch.Generator().manual_seed(self.seed)
        self.queries = None
        self.corpus = None
        self.train_loader = None
        self.dev_loader = None
        self.init()

    def get_dataset_name(self):
        return ".".join(self.corpus_path.split("/")[-1].split(".")[:-1])

    def _preprocess(self,queries,corpus):
        norm_text = lambda s: s.replace(" 's","'s").replace(" 'nt"," not")
        return {k: v | {"query":norm_text(v["query"])} for k,v in queries.items()}, {k:norm_text(v) for k,v in corpus.items()}

    def _load_data(self):
        if self.corpus is None or self.queries is None:
            self.corpus = {}
            with open(self.corpus_path) as f:
                self.corpus = json.load(f)
            self.queries = {}
            with open(self.query_path) as f:
                for k,v in json.load(f).items():
                    if self.task=="evaluate" or (self.task=="best_fit" and ((len(v["hard_neg"])>=self.hard_neg_count) if self.hard_neg_count else (len(v["hard_neg"])==5))) or (self.task=="rank" and len(v["qrels"])>=self.passage_count):
                        self.queries[k] = v
        return self.queries,self.corpus

    def get_loaders(self):
        return self.train_loader,self.dev_loader

    def init(self):
        queries,corpus = self._preprocess(*self._load_data())

        full_dataset = RawDataset(self.task,queries, corpus,hard_neg_count=self.hard_neg_count,passage_count=self.passage_count)
        pad = lambda x: x+[0 for _ in range(3-len(x))]
        train_dataset,_,dev_dataset = torch.utils.data.random_split(full_dataset,pad(self.splits[:1]+[len(full_dataset)-sum(self.splits)]+self.splits[1:]),generator=self.generator)

        print("Dataset:",self.get_dataset_name(),"Full Queries:",len(full_dataset),"Train Queries:",len(train_dataset),"Dev Queries:",len(dev_dataset))

        if self.with_idf:
            passages_train = [passage for qps,_ in train_dataset for passage in qps[1:]]
            collate_train = CollateBERTIDF(self.pretrained_model_name,frozenset(passages_train),self.idf_max_features, self.idf_pca_components,self.seed)
            passages_dev = [passage for qps,_ in dev_dataset for passage in qps[1:]]
            collate_dev = None if len(passages_dev)==0 else CollateBERTIDF(self.pretrained_model_name,frozenset(passages_dev),self.idf_max_features, self.idf_pca_components,self.seed)
        else:
            collate_train = CollateBERT(self.pretrained_model_name)
            collate_dev = CollateBERT(self.pretrained_model_name)

        self.train_loader = DataLoader(train_dataset, batch_size=self.mini_batch_size, shuffle=True, collate_fn=collate_train,generator=self.generator, num_workers=2)
        if len(dev_dataset)>=1:
            self.dev_loader = DataLoader(dev_dataset, batch_size=self.mini_batch_size, shuffle=True, collate_fn=collate_dev,generator=self.generator, num_workers=2)

        train_dataset_size = len(self.train_loader)
        print(f"Size of the train dataset: {train_dataset_size}")
        if len(dev_dataset) >= 1:
            dev_dataset_size = len(self.dev_loader)
            print(f"Size of the val dataset: {dev_dataset_size}")


class RawDataset(Dataset):
    def __init__(self, task, queries, corpus,hard_neg_count=None,passage_count=None):
        self.task = task
        self.queries = queries
        self.corpus = corpus
        self.passage_count = passage_count
        self.hard_neg_count = hard_neg_count
        self.ids = list(self.queries.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        anchor = self.queries[self.ids[index]]["query"]

        if self.task=="evaluate":
            random.shuffle(self.queries[self.ids[index]]["qrels"])
            passages,labels = zip(*((self.corpus[pid],relevance) for pid,relevance in self.queries[self.ids[index]]["qrels"]))
            return [anchor]+list(passages),list(labels)

        if self.task=="best_fit":
            positive = self.corpus[self.queries[self.ids[index]]["pos"][0]]
            hard_negatives = [self.corpus[pid] for pid in self.queries[self.ids[index]]["hard_neg"][:self.hard_neg_count]]
            return [anchor,positive]+hard_negatives,None

        if self.task=="rank":
            random.shuffle(self.queries[self.ids[index]]["qrels"])
            passages,labels = zip(*((self.corpus[pid],relevance) for pid,relevance in self.queries[self.ids[index]]["qrels"][:self.passage_count]))
            return [anchor]+list(passages),list(labels)
        return None


class BertEncoder:
    def __init__(self, pretrained_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def encode(self,inputs):
        encode = lambda text: self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=256, truncation=True, pad_to_max_length=True,return_attention_mask=True,return_tensors = 'pt')
        input_ids, attention_mask = zip(*((rt["input_ids"],rt["attention_mask"]) for text in inputs if (rt:=encode(text))))
        return torch.stack((torch.stack(input_ids),torch.stack(attention_mask)))


class CollateBERT:
    def __init__(self,pretrained_model_name):
        self.bert_encoder = BertEncoder(pretrained_model_name)

    def __call__(self, batch):
        bert_batch,label_batch = zip(*((self.bert_encoder.encode(inputs[0]),inputs[1]) for inputs in batch))
        if label_batch[0] is None:
            return torch.stack(bert_batch)
        else:
            return torch.stack(bert_batch), torch.tensor(label_batch).float()


class CollateBERTIDF:
    def __init__(self,pretrained_model_name,corpus,max_features,pca_components,seed):
        self.bert_encoder = BertEncoder(pretrained_model_name)
        self.tfidf_encoder = IDFEncoder(corpus,max_features,pca_components,seed)

    def __call__(self, batch):
        bert_batch_tfidf_batch,label_batch = zip(*(((self.bert_encoder.encode(inputs[0]),self.tfidf_encoder.encode(inputs[0])),inputs[1]) for inputs in batch))
        bert_batch,tfidf_batch = zip(*bert_batch_tfidf_batch)
        if label_batch[0] is None:
            return torch.stack(bert_batch),torch.stack(tfidf_batch)
        else:
            return (torch.stack(bert_batch),torch.stack(tfidf_batch)), torch.tensor(label_batch).float()


class IDFEncoder:

    def __init__(self, corpus,max_features,pca_components,seed):
        self.tokenizer = Tokenizer(lang="en")
        self.pca_components = pca_components
        self.vectorizer = TfidfVectorizer(analyzer="word",stop_words='english',binary=True,norm='l2',max_features=max_features)
        self.svd = TruncatedSVD(n_components=pca_components,random_state=seed)
        self.svd.fit_transform(self.vectorizer.fit_transform(self.tokenize(corpus)))


    def encode(self,inputs):
        pad = lambda vec: vec+[0.0 for _ in range(self.pca_components-len(vec))]
        return tensor(list(map(lambda vec: pad(vec.tolist()),self.svd.transform(self.vectorizer.transform(self.tokenize(inputs))))))

    def tokenize(self,texts):
        return [" ".join(self.tokenizer.stem(token) for token in self.tokenizer.tokenize(text.lower().strip())) for text in texts]


class Tokenizer:
    def __init__(self, stop_words=[], lang=None):
        ALL_PUNCTUATIONS = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        if lang=="de":
            from spacy.lang.de.stop_words import STOP_WORDS
            from spacy.lang.de import German
            self.nlp = German()
            self.stemmer = SnowballStemmer(language='german')
            self.stemmer.stemmer._GermanStemmer__step2_suffixes += ("in",)
            self.stop_words = list(ALL_PUNCTUATIONS) + ([] if stop_words is None else stop_words+list(STOP_WORDS) )
        else:
            from spacy.lang.en.stop_words import STOP_WORDS
            from spacy.lang.en import English

            self.nlp = English()
            self.stemmer = SnowballStemmer(language='english')
            self.stop_words = list(ALL_PUNCTUATIONS) + ([] if stop_words is None else stop_words+list(STOP_WORDS) )
        self.lang = lang

    def tokenize(self, text):
        return [t.text for t in self.nlp.tokenizer(text) if len(t.text.strip())>0 and t.text.lower() not in self.stop_words]

    @functools.lru_cache(maxsize=1000)
    def stem(self,word):
        if self.lang=="de":
            return self.stemmer.stem(self.stemmer.stem(word))
        else:
            return self.stemmer.stem(word)
