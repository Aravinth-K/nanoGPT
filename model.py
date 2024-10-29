"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

def get_norm(config):
    norm_type = config.norm_type.lower()
    if norm_type == 'layernorm':
        return LayerNorm(config.n_embd, config.bias)
    elif norm_type == 'rmsnorm':
        return RMSNorm(config.n_embd, config.bias)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    
def get_activation(activation_name):
    activation_name = activation_name.lower()
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'silu':
        return nn.SiLU()
    elif activation_name == 'selu':
        return nn.SELU()
    elif activation_name == 'solu':
        return SoLU()  # Anthropic (primarily for interpretability)
    if activation_name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")
    
class SoLU(nn.Module):
    def forward(self, x):
        return x * nn.Softmax()(x)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class RMSNorm(nn.Module):
    def __init__(self, ndim, bias=True, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt()
        if self.bias is not None:
            return self.scale * x / (norm + self.eps) + self.bias
        else:
            return self.scale * x / (norm + self.eps)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.rope = config.rope
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        if self.rope:
            self.rotary_emb = RoPE(config.n_head)

    def forward(self, x, pos=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rope and pos is not None:
            q, k = self.rotary_emb.apply_rotary_pos_emb(q, k, pos)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = get_activation(config.activation)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class FeedForward(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * int(2 * config.n_embd / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        print(f"FeedForward hidden_dim: {hidden_dim}")
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = get_activation(config.activation)

    def forward(self, x):
        return self.dropout(self.w2(self.activation(self.w1(x)) * self.w3(x)))

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = get_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = get_norm(config)
        if config.mlp == "gpt":
            self.mlp = MLP(config)
        elif config.mlp == "llama":
            self.mlp = FeedForward(config)
        else:
            raise ValueError(f"Unsupported MLP type: {config.mlp}")

    def forward(self, x, pos=None):
        x = x + self.attn(self.ln_1(x), pos)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class RoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def apply_rotary_pos_emb(self, q, k, pos):
        sinusoid_inp = torch.einsum("i , j -> i j", pos, self.inv_freq)
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot

    @staticmethod
    def rotate_half(x):
        x = x.view(*x.shape[:-1], -1, 2)
        x = torch.stack([-x[..., 1], x[..., 0]], dim=-1)
        return x.flatten(-2)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, n_embd, max_seq_len=512):
        super(PositionalEmbedding, self).__init__()

        self.n_embd = n_embd
        self.max_seq_len = max_seq_len

        inv_freq = 1 / (10000 ** (torch.arange(0.0, n_embd, 2.0) / n_embd))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        # pos_seq: [t]
        # Ensure pos_seq is float for torch.outer
        pos_seq = pos_seq.float()
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)  # [t, n_embd/2]
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)  # [t, n_embd]
        pos_emb = pos_emb.unsqueeze(0)  # [1, t, n_embd]
        return pos_emb

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    norm_type: str = 'layernorm'
    mlp: str = 'gpt'
    activation: str = 'gelu'
    pos_emb: str = 'learned'
    rope: bool = False
    num_targets: int = 2
    intermediate_loss: bool = False
    max_iters: int = 600000
    weight_tying: bool = True

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = get_norm(config),
        ))

        # Positional Embeddings
        if config.pos_emb == 'absolute':
            self.transformer.wpe = PositionalEmbedding(config.n_embd)
        elif config.pos_emb == 'learned':
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_emb == 'learned_per_layer':
            self.transformer.wpe = nn.ModuleList([
                nn.Embedding(config.block_size, config.n_embd) for _ in range(config.n_layer)
            ])
        else:
            raise ValueError(f"Unsupported pos_emb: {config.pos_emb}")
        
        if config.intermediate_loss:
            # Create a separate auxiliary head for each target per layer
            self.aux_lm_heads = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
                    for _ in range(config.num_targets)
                ]) 
                for _ in range(config.n_layer)
            ])
        
        # Define multiple LM heads
        self.lm_heads = nn.ModuleList([
            nn.Linear(config.n_embd, config.vocab_size, bias=config.bias) 
            for _ in range(config.num_targets)
        ])
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if config.weight_tying:
            self.transformer.wte.weight = self.lm_heads[0].weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if self.config.pos_emb == 'learned_per_layer':
                for i in range(self.config.n_layer):
                    n_params -= self.transformer.wpe[i].weight.numel()
            else:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

    def forward(self, idx, targets=None, eval_only=False, current_iter=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        # Positional embeddings
        if self.config.pos_emb == 'learned':
            pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        elif self.config.pos_emb == 'learned_per_layer':
            x = self.transformer.drop(tok_emb)
        else:
            raise ValueError(f"Unsupported pos_emb: {self.config.pos_emb}")
        
        total_loss = 0

        for layer_idx, block in enumerate(self.transformer.h):
            if self.config.pos_emb == 'learned_per_layer':
                pos_emb_layer = self.transformer.wpe[layer_idx](pos)
                x = self.transformer.drop(x + pos_emb_layer)
            x = block(x, pos if self.config.rope else None)

            # Compute auxiliary predictions from the current layer
            if self.config.intermediate_loss and targets is not None and not eval_only:
                # Calculate the dropout step for this layer
                dropout_step = (layer_idx + 1) / (2 * self.config.n_layer) * self.config.max_iters

                intermediate_losses = []
                if current_iter is not None and current_iter < dropout_step:
                    for k in range(self.config.num_targets):
                        if k == 0:
                            shifted_target = targets.contiguous()
                            aux_logits = self.aux_lm_heads[layer_idx][k](x).contiguous()
                        else:
                            # Shift targets for each additional target
                            shifted_target = targets[:, k:].contiguous()
                            # Adjust logits to align with shifted targets
                            aux_logits = self.aux_lm_heads[layer_idx][k](x)[:, :-k, :].contiguous()
                        # Compute cross-entropy loss
                        aux_loss = F.cross_entropy(
                            aux_logits.view(-1, aux_logits.size(-1)),
                            shifted_target.view(-1),
                            ignore_index=-1
                        )
                        # Weight intermediate losses less than the main loss
                        weight = (0.5) ** k
                        intermediate_losses.append(weight * aux_loss)
                total_loss += sum(intermediate_losses)

        x = self.transformer.ln_f(x)

        if targets is not None and not eval_only:
            # Generate logits for all targets
            logits = [head(x) for head in self.lm_heads]  # List of tensors: [ (b, t, vocab), ... ]

            # Prepare shifted targets for each head
            # Head 0 predicts t_i given t_{0:i-1}
            # Head 1 predicts t_{i+1} given t_{0:i-1}, etc.
            shifted_targets = []
            for k in range(self.config.num_targets):
                if k == 0:
                    target = targets.contiguous()
                else:
                    target = targets[:, k:].contiguous()
                    logits[k] = logits[k][:, :-k, :].contiguous()
                shifted_targets.append(target)

            # Compute losses for each head
            losses = []
            for k in range(self.config.num_targets):
                loss = F.cross_entropy(
                    logits[k].view(-1, logits[k].size(-1)),
                    shifted_targets[k].view(-1),
                    ignore_index=-1
                )
                # Apply weighting: primary loss has weight 1.0, auxiliary losses have weight 0.5
                weight = (0.5) ** k
                losses.append(weight * loss)

            # Sum all losses
            total_loss += sum(losses)
        elif targets is not None and eval_only:
            logits = self.lm_heads[0](x)
            total_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_heads[0](x[:, [-1], :]) # note: using list [-1] to preserve the time dim; (b, 1, vocab)
            total_loss = None

        return logits, total_loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
