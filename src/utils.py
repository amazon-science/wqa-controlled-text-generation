from tkinter import S
import numpy as np
import random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import trange
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import transformers
from transformers import GPT2Model, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import pprint
import logging

pp = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(700)

class RTPDataset(Dataset):
    def __init__(self, fname, p=None):
      self.data = self.load_data(fname)
      self.texts = list(self.data.prompt_text.values) + list(self.data.continuation_text.values)
      self.len = len(self.texts)
      if p is not None:
        self.len = int(self.len*p) 
        self.texts = self.texts[:self.len]     
      print(f"Loading {self.len} text phrases...")
 
    def load_data(self, fname):
      data = pd.read_csv(fname)
      return data  

    def __len__(self):
      return self.len

    def __getitem__(self, idx):
      return self.texts[idx]

class RSPDataset(Dataset):
    def __init__(self, fname, p=None):
      self.data = self.load_data(fname)
      self.texts = list(self.data.prompt.values) + list(self.data.continuation.values)
      self.len = len(self.texts)
      if p is not None:
        self.len = int(self.len*p) 
        self.texts = self.texts[:self.len]     
      print(f"Loading {self.len} text phrases...")
 
    def load_data(self, fname):
      data = pd.read_csv(fname)
      return data  

    def __len__(self):
      return self.len

    def __getitem__(self, idx):
      return self.texts[idx]   

class PWKPDataset(Dataset):
    def __init__(self, fname, p=None):
      self.texts = self.load_data(fname)
      self.len = len(self.texts)
      if p is not None:
        self.len = int(self.len*p) 
        self.texts = self.texts[:self.len]     
      print(f"Loading {self.len} text phrases...")
 
    def load_data(self, fname):
      with open(fname, "r") as handle:
        text = [t.strip() for t in handle.readlines()]
      return text  

    def __len__(self):
      return self.len

    def __getitem__(self, idx):
      return self.texts[idx]            
