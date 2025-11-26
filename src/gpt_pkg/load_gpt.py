import torch
import numpy as np

import pickle
import tiktoken
import pytz
import sys
import os

from datetime import datetime
from contextlib import nullcontext

from gpt_pkg.model import GPTConfig, GPT

def load_transformer_model(model_name, device, dataset, checkpoint_dir, data_pkg_dir, compile = False):

    def check_string(s):
        if "nuc_extended" in s:
            return "nuc_extended"
        elif "amino" in s:
            return "amino"
        elif "nuc" in s:
            return "nuc"
        else:
            return None
        
    sequence_type = check_string(dataset)

    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')

    seed            = 1337
    dtype           = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    #compile        =  False # use PyTorch 2.0 to compile the model to be faster

    # -----------------------------------------------------------------------------
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32       = True # allow tf32 on cudnn
    device_type                           = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype                               = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path  = os.path.join(checkpoint_dir, model_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
        best_val_loss = checkpoint['best_val_loss']
        print('best_val_loss: ', best_val_loss)

        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict,strict = False) # Build in method allows to load models parameters from dictionary state_dicts, https://github.com/CannyLab/summary_loop/issues/3

    
    model.eval()
    model.to(device)

    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
        # look for the meta pickle in case it is available in the dataset folder

        
    meta_path = os.path.join(data_pkg_dir, f'meta_{sequence_type}.pkl') # stores vocabulary mappings i.e. {'A': 0, 'C': 1, 'T': 2, 'G': 3} and {0: 'A', 1: 'C', 2: 'T', 3: 'G'} needed for tokenization
    load_meta = os.path.exists(meta_path)
    print('load_meta: ', load_meta)

    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        vocab_size = len(itos)
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])

    else:
        print("ERROR - No meta.pkl found.")
        sys.exit()
        
    return model, encode, decode, vocab_size, ctx # Return model, encode function (text to tokens) decode function (tokens to text), vocab size and torch.autocast context for mixed-precision inference which automatically selects the appropriate precision (e.g., float16, bfloat16, or float32) for different operations to improve performance and memory efficiency.


if __name__ == '__main__':

    print('load_gpt.py')
   