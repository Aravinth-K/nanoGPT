# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-ewik8-char'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'enwik8-char'
wandb_run_name = 'mini-gpt'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 32 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 2
n_embd = 512
dropout = 0.2
norm_type = 'layernorm'
mlp = 'gpt'
activation = 'relu'
pos_emb = 'learned_per_layer'
rope = False
bias = True
residual_attention = False
residual_attention_mode = 'add'
pre_ln = True
in_gate = True

num_targets = 2
intermediate_loss = True
learning_rate = 3e-3 # with baby networks can afford to go a bit higher
max_iters = 1000
lr_decay_iters = 1000 # make equal to max_iters usually
min_lr = 3e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

adaptive_span = True
adapt_span_loss_coeff = 0.000002
ramp_size = 32

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
