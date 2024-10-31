out_dir = 'out-ewik8-char'
eval_interval = 1000
eval_iters = 200
final_eval_iters = 2000
log_interval = 100

always_save_checkpoint = False

wandb_log = True
wandb_project = 'enwik8-char'
wandb_run_name = 'mini-gpt'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 512 

n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.1
norm_type = 'rmsnorm'
mlp = 'glu'
activation = 'silu'
pos_emb = 'learned_per_layer'
rope = False
bias = True

adaptive_span= True
adapt_span_loss_coeff= 0.000002
ramp_size= 32

num_targets = 2
intermediate_loss = True
learning_rate = 3e-3
max_iters = 10000
lr_decay_iters = 10000 
min_lr = 3e-4 # learning_rate / 10 usually

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cuda'  # run on cpu only
compile = True # do not torch compile the model
