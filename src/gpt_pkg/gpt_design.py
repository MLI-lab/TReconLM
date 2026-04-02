# %%
from gpt_pkg.model import GPTConfig, GPT

# %%
gpt_models = {
    'gpt_A': dict(n_layer=4, n_head=8, n_embd=256),    # ~3
    'gpt_B': dict(n_layer=6, n_head=6,   n_embd=384),  # ~10M params
    'gpt_C': dict(n_layer=12, n_head=6,  n_embd=384),  # ~21M params
    'gpt_D': dict(n_layer=12, n_head=8,  n_embd=512),  # ~37M params
    'gpt_E': dict(n_layer=12, n_head=12, n_embd=768),  # ~85M params 
    'gpt_F': dict(n_layer=24, n_head=16, n_embd=768),  # ~170M params 
    'gpt_G': dict(n_layer=24, n_head=16, n_embd=1024), # ~300M params
    'gpt_H': dict(n_layer=36, n_head=20, n_embd=1280), # ~710M params
    'gpt_I': dict(n_layer=48, n_head=25, n_embd=1600), # ~1.5B params
}

# %%
selected_model = 'gpt_I'

selected_model_values = gpt_models[selected_model]

n_layer = selected_model_values['n_layer']
n_head = selected_model_values['n_head']
n_embd = selected_model_values['n_embd']

print(f"Selected model: {selected_model}")
print(f"n_layer: {n_layer}, n_head: {n_head}, n_embd: {n_embd}")

dropout  = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
bias     = None # do we use bias inside LayerNorm and Linear layers?
block_size = 600
label_smoothing = 0
vocab_size = 8 # A,C,G,T,:,|,-,#

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
            bias=bias, vocab_size=vocab_size, dropout=dropout, label_smoothing = label_smoothing) 

# gpt model
gpt_cfg = GPTConfig(**model_args)
model = GPT(gpt_cfg)

num_params = model.get_num_params(non_embedding=True)
print('num_params:', num_params)

approx_num_params = 12*n_layer*n_embd**2 
print('approx_num_params:', approx_num_params)

abs_delta = abs(num_params - approx_num_params)
rel_delta = abs_delta/num_params
print('rel_delta:', rel_delta)



