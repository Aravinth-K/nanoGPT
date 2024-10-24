out_dir = 'out-ewik8-char'
eval_interval = 250
eval_iters = 200
log_interval = 10 

always_save_checkpoint = False

wandb_log = True
wandb_project = 'enwik8-char'
wandb_run_name = 'mini-gpt'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 512 

n_layer = 12
n_head = 2
n_embd = 512
dropout = 0.2
norm_type = 'layernorm'
mlp = 'gpt'
activation = 'relu'
pos_emb = 'learned_per_layer'
rope = False

num_targets = 2
intermediate_loss = True
learning_rate = 3e-3
max_iters = 10000
lr_decay_iters = 10000 
min_lr = 3e-4 # learning_rate / 10 usually
beta2 = 0.99 

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cuda'  # run on cpu only
compile = True # do not torch compile the model
