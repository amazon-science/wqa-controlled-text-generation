import os
import json
from tqdm import tqdm
import argparse
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import gc
from model import GPT2TransformedLMHeadModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd

gc.collect()
torch.cuda.empty_cache()

class GPTCollate(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, sequences):
        if self.tokenizer.bos_token:
            sequences = [self.tokenizer.bos_token+s for s in sequences]        
        tokens = self.tokenizer(
            sequences, 
            padding=True, 
            return_tensors="pt"
        )
        return  tokens.input_ids, tokens.attention_mask, sequences


class RealAttributePromptsDataset(Dataset):
    def __init__(self, prompts_path, n_gens=1):
      if n_gens > 1:
        print(f"Generating {n_gens} continuations per prompt!")
      self.prompts = self.load_prompts(prompts_path)
      self.prompts = [p for p in self.prompts for _ in range(n_gens)]
      self.prompts = [p.encode('ASCII', 'ignore').decode(encoding="utf-8").strip() for p in self.prompts]
      print(f"Loaded {len(self.prompts)} prompts!")

    def load_prompts(self, fname):
      if fname.endswith(".txt"):
        return self.load_txt(fname)
      return self.load_csv(fname)

    def load_txt(self, fname):
        with open(fname, "r") as handle:
            prompts = handle.readlines()
        return prompts      

    def load_csv(self, fname):
      data = pd.read_csv(fname)
      promps = list(data.prompt_text.values)
      return promps    

    def __len__(self):
      return len(self.prompts)

    def __getitem__(self, idx):
      return self.prompts[idx]

def get_data_loader(prompt_path, tokenizer, n_gens, batch_size):
    collate_fn = GPTCollate(tokenizer)
    rtpdataset = RealAttributePromptsDataset(prompt_path, n_gens)
    rtpdataloader = DataLoader(
        rtpdataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    return rtpdataloader


def generate(
    model, 
    tokenizer, 
    dataloader, 
    device,
    n_tokens_gen=25, 
    top_p=0.8,
    repetition_penalty=1.2,
    verbose=False,
    debug=False
    ):
    generations = []
    for i, (input_ids, attention_masks, prompt) in enumerate(tqdm(dataloader, total=len(dataloader))):

        if i > 5 and debug:
            break

        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        position_ids = attention_masks.cumsum(dim=1) - 1
        
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_masks,
            position_ids=position_ids,
            do_sample=True, 
            max_new_tokens=n_tokens_gen,            
            pad_token_id=tokenizer.eos_token_id,
            top_p=top_p, 
            repetition_penalty=repetition_penalty,
        )

        generation = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if verbose:
            for p, g in zip(prompt, generation):
                print(p)
                print(g, "\n")

        generations += generation
    
    return generations


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the transform head.')
    parser.add_argument('--prompt_path', type=str, help="Path to generate csv for proompts")
    parser.add_argument('--model_type', default="gpt2", type=str, help="Model type")
    parser.add_argument('--base_model_type', default="gpt2-medium", type=str, help="Model type")
    parser.add_argument('--chrt_model_path', type=str, default="gpt2", help="Finetuned CHRT model path")
    parser.add_argument('--generations_output_dir', type=str, help="Output dir to save generated text.")
    parser.add_argument('--generations_output_fname', type=str, help="Output fname to save generated text.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for generating continuations.")
    parser.add_argument('--n_gens', type=int, default=25, help="Number of generations per prompt.")
    parser.add_argument('--n_tokens_gen', type=int, default=25, help="Number of tokens to generate.")
    parser.add_argument('--top_p', type=float, default=0.8, help="Top p for nucleus sampling.")
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help="Reptition Penalty.")
    parser.add_argument('--verbose', action='store_true', help="Verbose will print each generation.")
    parser.add_argument('--debug', action='store_true', help="debug will only run 5 forwrad passes.")
    parser.add_argument('--n_blocks', type=int, default=2, help="Number of transform blocks.")
    parser.add_argument('--factor', type=float, default=0.5, help="Factor of hidden state for each transform block.")
    parser.add_argument('--p_dropout', type=float, default=0.1, help="Dropout prob for each transform block.")    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_type, add_prefix_space=True)
    device = "cuda"

    if args.model_type == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = GPT2TransformedLMHeadModel.from_pretrained(
            args.base_model_type, 
            is_train=False,
            n_blocks=args.n_blocks, 
            factor=args.factor, 
            p_dropout=args.p_dropout
        ).to(device)
    model.load_state_dict(torch.load(args.chrt_model_path))
    model.eval()
    dataloader = get_data_loader(args.prompt_path, tokenizer, args.n_gens, args.batch_size)
    
    generations = generate(
        model, 
        tokenizer, 
        dataloader, 
        device, 
        n_tokens_gen=args.n_tokens_gen, 
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        verbose=args.verbose,
        debug=args.debug
    )
    
    generations_output_dir = args.generations_output_dir
    generations_output_fname = args.generations_output_fname

    if not os.path.exists(generations_output_dir):
        os.makedirs(generations_output_dir)

    generations_output_path = os.path.join(generations_output_dir, generations_output_fname)

    with open(generations_output_path, "w") as handle:
        handle.write(
            "\n".join([json.dumps(generation) for generation in generations])
        )