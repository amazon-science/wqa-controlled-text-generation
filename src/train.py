import torch.nn as nn
import argparse
import random, os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from utils import RTPDataset, RSPDataset, PWKPDataset
import gc
from model import GPT2TransformedLMHeadModel
import wandb
wandb.init(
  project="<EXPERIMENT_NAME>",
)

def normalize_weights(W_TRIP, W_OG):
    """Make the sum of the weights 1"""
    W_SUM = W_TRIP + W_OG
    W_TRIP = W_TRIP / W_SUM
    W_OG = W_OG / W_SUM
    return W_TRIP, W_OG

def seed_everything(seed):
    """Set random seed for everything."""
    print(f"Seeding everything with seed: {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(700)

def count_params(m):
    """Count number of trainable parameters in a pytorch nodel"""
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def freeze_weights(m):
    """freeze all the weights for a pytorch model"""
    for param in m.parameters():
        param.requires_grad = False

def get_tokenizer(name="gpt2"):
    """get GPT2 tokenizer"""
    tokenizer =  GPT2Tokenizer.from_pretrained(name, add_prefix_space=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class GPTCollate(object):
    """collate function for gpt2. 
    Convert the text sequence into sequence of input ids and attention masks."""
    def __init__(self, tokenizer):
        self.tokenizer  = tokenizer

    def __call__(self, sequences):
        tokens = self.tokenizer(
            [self.tokenizer.bos_token+s for s in sequences], 
            padding=True, 
            return_tensors="pt"
        )
        return  tokens.input_ids, tokens.attention_mask

def get_rtp_loaders(tokenizer, train_path, test_path, bs, p=None):
    """Get dataloaders for the RealToxicityPrompts dataset"""
    collate_fn = GPTCollate(tokenizer)

    train_dataset = RTPDataset(train_path, p=p)
    test_dataset = RTPDataset(test_path, p=p)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_dataloader, test_dataloader, collate_fn

def get_rsp_loaders(tokenizer, train_path, test_path, bs, p=None):
    """Get dataloaders for the RealSentimentPrompts dataset"""
    collate_fn = GPTCollate(tokenizer)

    train_dataset = RSPDataset(train_path, p=p)
    test_dataset = RSPDataset(test_path, p=p)


    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_dataloader, test_dataloader, collate_fn   

def get_ygd_loaders(tokenizer, train_path, test_path, bs, p=None):
    """Get dataloaders for the YelpGenderPDataset"""
    collate_fn = GPTCollate(tokenizer)

    train_dataset = PWKPDataset(train_path, p=p)
    test_dataset = PWKPDataset(test_path, p=p)


    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_dataloader, test_dataloader, collate_fn        


@torch.no_grad()
def get_h(m, input_ids, attention_mask, position_ids):
    return m(
        input_ids, 
        attention_mask=attention_mask, 
        position_ids=position_ids,
        output_hidden_states=True,
        ).hidden_states[-1]

def train(loader, model, model_pos, model_neg, optimizer, criterion_preservation, criterion_contrast, device="cuda", scheduler=None, type="Train"):
  model.train()
  running_losses = {
    f"{type} loss": 0,
    f"{type} loss_triplet": 0,
    f"{type} loss_og": 0,
  }  
  l = len(loader)

  for (inp) in tqdm(loader):
    optimizer.zero_grad()

    if isinstance(inp, tuple):
        input_ids, attention_mask = inp
        position_ids = attention_mask.cumsum(dim=1) - 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)        
    else:
        input_ids = inp
        attention_mask = None
        position_ids = None
        input_ids = input_ids.to(device)    
   

    with torch.no_grad(): #extra safety xD
        h_pos = get_h(model_pos, input_ids, attention_mask, position_ids)
        h_neg = get_h(model_neg, input_ids, attention_mask, position_ids)

    out = model(
        input_ids, 
        attention_mask=attention_mask, 
        position_ids=position_ids,
        output_hidden_states=True,
    )
    h_og = out.hidden_states[0][-1]    
    h_t = out.hidden_states[-1]
    loss_og = criterion_preservation(h_t, h_og)
    loss_triplet = criterion_contrast(h_t, h_pos, h_neg)
    loss = W_OG*loss_og + W_TRIP*loss_triplet


    # loss_neg = criterion(h_t, h_neg)

    # # loss = loss_og + loss_pos + loss_neg
    # loss = loss_pos - loss_neg
    loss.backward()
    
    optimizer.step()
    if scheduler:
        scheduler.step()



    with torch.no_grad():
        running_losses[f"{type} loss"] += loss.detach().cpu().numpy().tolist()/l
        running_losses[f"{type} loss_triplet"] += loss_triplet.detach().cpu().numpy().tolist()/l
        running_losses[f"{type} loss_og"] += loss_og.detach().cpu().numpy().tolist()/l
        # running_losses["loss_neg"] += loss_neg.detach().cpu().numpy().tolist()/l   
        #    

    loss.detach()
    loss_og.detach()
    loss_triplet.detach()

    gc.collect()

  return running_losses


@torch.no_grad()
def evaluate(loader, model, model_pos, model_neg, criterion_preservation, criterion_contrast, device="cuda", type="Train/Test"):
  model.eval()
  running_losses = {
    f"{type} loss": 0,
    f"{type} loss_triplet": 0,
    f"{type} loss_og": 0,
    f"{type} loss_pos": 0,
    f"{type} loss_neg": 0,
  }  
  l = len(loader)
  for (inp) in tqdm(loader):

    if isinstance(inp, tuple):
        input_ids, attention_mask = inp
        position_ids = attention_mask.cumsum(dim=1) - 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)        
    else:
        input_ids = inp
        attention_mask = None
        position_ids = None
        input_ids = input_ids.to(device)  

    h_pos = get_h(model_pos, input_ids, attention_mask, position_ids)
    h_neg = get_h(model_neg, input_ids, attention_mask, position_ids)
    out = model(
        input_ids, 
        attention_mask=attention_mask, 
        position_ids=position_ids,
        output_hidden_states=True,
    )
    h_og = out.hidden_states[0][-1]    
    h_t = out.hidden_states[-1]

    loss_og = criterion_preservation(h_t, h_og)
    loss_triplet = criterion_contrast(h_t, h_pos, h_neg)

    loss_pos = criterion_preservation(h_t, h_pos)
    loss_neg = -1*criterion_preservation(h_t, h_neg)
    loss = W_TRIP*loss_triplet + W_OG*loss_og

    running_losses[f"{type} loss"] += loss.detach().cpu().numpy().tolist()/l
    running_losses[f"{type} loss_triplet"] += loss_triplet.detach().cpu().numpy().tolist()/l
    running_losses[f"{type} loss_og"] += loss_og.detach().cpu().numpy().tolist()/l
    running_losses[f"{type} loss_pos"] += loss_pos.detach().cpu().numpy().tolist()/l
    running_losses[f"{type} loss_neg"] += loss_neg.detach().cpu().numpy().tolist()/l  

  return running_losses

def get_optim_scheduler(net, train_loader, EPOCHS):
    optimizer = optim.AdamW(
        net.parameters(), 
        lr = 2e-5, 
        eps = 1e-8 
    )
    num_train_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_train_steps * 0.1) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps, 
        num_training_steps = num_train_steps            
    )
    # scheduler = None
    return optimizer, scheduler  


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

def load_models(model_path_or_name, positive_model_path, negative_nodel_path, device, n_blocks=2, factor=0.5, p_dropout=0.1):
    model = GPT2TransformedLMHeadModel.from_pretrained(
        model_path_or_name, 
        n_blocks=n_blocks, 
        factor=factor, 
        p_dropout=p_dropout
    ).to(device)

    positive_model = GPT2LMHeadModel.from_pretrained(positive_model_path).to(device)
    negative_model = GPT2LMHeadModel.from_pretrained(negative_nodel_path).to(device)
    return model, positive_model, negative_model

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the transform head.')
    parser.add_argument('--task', type=str, default="toxicity", help="Attribute task.")
    parser.add_argument('--train_path', type=str, default=None, help="Training data path")
    parser.add_argument('--test_path', type=str, default=None, help="Test data path")
    parser.add_argument('--model_type', default="gpt2", type=str, help="Model type")
    parser.add_argument('--model_path_or_name', type=str, default="gpt2", help="Finetuned B model path")
    parser.add_argument('--positive_model_path', type=str, help="Finetuned positive model path")
    parser.add_argument('--negative_nodel_path', type=str, help="Finetuned negative model path")
    parser.add_argument('--model_output_dir', type=str, help="Output dir to save models.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training transform head.")
    parser.add_argument('--n_epochs', type=int, default=8, help="Number of epochs for training transform head.")
    parser.add_argument('--triplet_weight', type=int, default=1, help="Weight for triplet loss.")
    parser.add_argument('--l2_weight', type=int, default=2, help="Weight for L2 loss.")
    parser.add_argument('--n_blocks', type=int, default=2, help="Number of transform blocks.")
    parser.add_argument('--factor', type=float, default=0.5, help="Factor of hidden state for each transform block.")
    parser.add_argument('--p_dropout', type=float, default=0.1, help="Dropout prob for each transform block.")
    parser.add_argument('--l2_loss_type', type=str, default="mse", help="Either mse or rmse")
    
    args = parser.parse_args()

    TASKS = ["toxicity", "sentiment", "pwkp"]
    MODEL_TYPES = ["gpt2"]

    #Parse Args
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    task = args.task
    model_type = args.model_type
    train_path = args.train_path
    test_path = args.test_path
    model_path_or_name = args.model_path_or_name
    positive_model_path = args.positive_model_path
    negative_nodel_path = args.negative_nodel_path
    model_output_dir = args.model_output_dir
    model_n_blocks = args.n_blocks
    model_factor = args.factor
    model_p_dropout = args.p_dropout
    l2_loss_type = args.l2_loss_type

    print(f"model_output_dir: {model_output_dir}")
    print(f"positive_model_path: {positive_model_path}")
    print(f"negative_nodel_path: {negative_nodel_path}")
    print(f"model_n_blocks: {model_n_blocks}")
    print(f"model_factor: {model_factor}")
    print(f"model_p_dropout: {model_p_dropout}")  

    #Some checks over args
    assert task in TASKS, f"{task} Not implemented."
    assert model_type in MODEL_TYPES, f"{model_type} Not implemented."

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    tokenizer = get_tokenizer(args.model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Let's use", torch.cuda.device_count(), " GPUs! Thank you Amazon!!")    

    if model_type == "gpt2":
        model, model_positive, model_negative = load_models(
            model_path_or_name, 
            positive_model_path, 
            negative_nodel_path, 
            device,
            n_blocks=model_n_blocks, 
            factor=model_factor, 
            p_dropout=model_p_dropout
        )
        model_positive.eval()
        model_negative.eval()

    assert (model_positive.lm_head.weight == model.lm_head.weight).all().detach().cpu().numpy(), "LM Head weights differ, we can only combine models with same LM head weights."
    assert (model_negative.lm_head.weight == model.lm_head.weight).all().detach().cpu().numpy(), "LM Head weights differ, we can only combine models with same LM head weights."
    print(f"#Weights before freezing: {count_params(model)}")
    model.freeze_except_transform_head()
    print(f"#Weights after freezing: {count_params(model)}")

    freeze_weights(model_positive)
    freeze_weights(model_negative)        

    if task == "toxicity":
        train_dataloader, test_dataloader, collate_fn = get_rtp_loaders(tokenizer, train_path, test_path, batch_size)
        # train_dataloader, test_dataloader = get_rtpfast_loaders(tokenizer, train_path, test_path, model_positive, model_negative, device, batch_size, p=None)
    if task == "sentiment":
        train_dataloader, test_dataloader, collate_fn = get_rsp_loaders(tokenizer, train_path, test_path, batch_size)
        # train_dataloader, test_dataloader = get_senti_loaders(tokenizer, train_path, test_path, block_size, batch_size)
    if task == "pwkp":
        train_dataloader, test_dataloader, collate_fn = get_ygd_loaders(tokenizer, train_path, test_path, batch_size)                

    criterion_contrast = nn.TripletMarginLoss(margin=1.0, p=2.0)
    if l2_loss_type == "mse":
        criterion_preservation = nn.MSELoss()
    else:
        criterion_preservation = RMSELoss()
    W_TRIP, W_OG = normalize_weights(args.triplet_weight, args.l2_weight)
    print(f"W_OG: {W_OG}")
    print(f"W_TRIP: {W_TRIP}")    

    gc.collect()

    #Train on Task
    if n_epochs:
        best_test_loss = np.inf
        optimizer, scheduler = get_optim_scheduler(model, train_dataloader, n_epochs)    
        best_output_model_path = os.path.join(model_output_dir, "best_model.pt")
        output_model_path = os.path.join(model_output_dir, "model.pt")        

        print(f"STARTING TRAINING FOR {task}...")
        for epoch in trange(n_epochs):
            print("="*10 + f" {epoch} " + "="*10)

            train_stats = train(train_dataloader, model, model_positive, model_negative, optimizer, criterion_preservation, criterion_contrast, device, scheduler, type=f"Train {task}")
            print(train_stats)

            test_stats = evaluate(test_dataloader, model, model_positive, model_negative, criterion_preservation, criterion_contrast, device, type="Test")
            print(test_stats)

            if test_stats['Test loss'] < best_test_loss:
                best_test_loss = test_stats['Test loss']  
                # torch.save(model.module.state_dict(), best_output_model_path)       
                torch.save(model.state_dict(), best_output_model_path)       

            wandb.log(train_stats)
            wandb.log(test_stats)  

        print(f"Best Test Loss: {best_test_loss}")
        # torch.save(model.module.state_dict(), output_model_path)
        torch.save(model.state_dict(), output_model_path)
