# rope 
out_dir = 'out-ewik8'
eval_interval = 1000
eval_iters = 200
final_eval_iters = 2000
log_interval = 100

always_save_checkpoint = False

wandb_log = True
wandb_project = 'enwik8'
wandb_run_name = 'initial_0'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 512 

n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.2
norm_type = 'layernorm'
pre_ln = True
mlp = 'gpt'
activation = 'relu'
pos_emb = 'none'
rope = True
bias = True
weight_tying = True

num_targets = 1
intermediate_loss = False

adaptive_span= False
adapt_span_loss_coeff= 0.000002
ramp_size= 32

num_memory_tokens: int = 0

residual_attention: bool = False  
residual_attention_mode: str = 'add'  

in_gate: bool = False

sparse_topk: int = 0

learning_rate = 3e-3
max_iters = 10000
lr_decay_iters = 10000 
min_lr = 3e-4 # learning_rate / 10 usually

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cuda'  # run on cpu only
compile = True # do not torch compile the model
