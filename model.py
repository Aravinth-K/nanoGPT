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
        
class AdaptiveSpan(nn.Module):
    """
    Implements the adaptive span mechanism from "Adaptive Attention Span in Transformers"
    """
    def __init__(self, 
                 max_span: int,
                 n_heads: int,
                 adapt_span_loss_coeff: float = 0.000002,
                 ramp_size: int = 32):
        """
        Parameters:
        -----------
        max_span: int
            Maximum attention span possible (S in paper)
        n_heads: int
            Number of attention heads
        adapt_span_loss_coeff: float
            Coefficient for the adaptive span loss (lambda in paper)
        ramp_size: int
            Size of the ramp for the masking function (R in paper)
        """
        super().__init__()
        
        self.max_span = max_span
        self.n_heads = n_heads
        self.adapt_span_loss_coeff = adapt_span_loss_coeff
        self.ramp_size = ramp_size
        
        # Initialize learnable spans for each head
        # z = Sz' where z' ∈ [0, 1] as per paper
        self.span_params = nn.Parameter(torch.zeros(n_heads))
        
        # Ensure spans start in [0, 1] range
        with torch.no_grad():
            self.span_params.data.clamp_(0, 1)

    def clamp_params(self):
        with torch.no_grad():
            self.span_params.data.clamp_(0, 1)
            
    def get_current_spans(self):
        """Returns current attention spans for all heads"""
        return self.max_span * self.span_params.clamp(0, 1)
    
    def get_loss(self):
        return self.adapt_span_loss_coeff * torch.sum(self.get_current_spans()) / self.n_heads
            
    def forward(self, attn):
        """
        Compute adaptive span mask and loss
        
        Parameters:
        -----------
        attn: torch.Tensor of shape (batch_size, n_heads, seq_len, seq_len)
            
        Returns:
        --------
        adaptive_mask: torch.Tensor of shape (batch_size, n_heads, seq_len, seq_len)
            Modified attention mask with learnable spans
        loss: torch.Tensor (scalar)
            L1 loss term for span sizes
        """
        seq_len = attn.size(-1)
        device = attn.device
        
        # Get current spans for each head
        current_spans = self.get_current_spans()  # shape: (n_heads,)

        # Create distance matrix
        positions = torch.arange(seq_len, device=device)
        # For each position i, calculate distance to previous positions j as (i - j)
        # This gives us how far back we're looking from the current position
        distances = positions.view(-1, 1) - positions.view(1, -1)  # shape: (seq_len, seq_len)
        distances = distances.abs()  # make distances positive

        # Expand dimensions for broadcasting
        distances = distances.view(1, 1, seq_len, seq_len)  # shape: (1, 1, seq_len, seq_len)
        current_spans = current_spans.view(-1, 1, 1)      # shape: (n_heads, 1, 1)
        
        # Compute soft_mask for all heads simultaneously
        # soft_mask = min(max((R + z - x)/R, 0), 1)
        # where R is ramp_size, z is current span, x is distance
        soft_mask = (self.ramp_size + current_spans - distances) / self.ramp_size  # shape: (n_heads, 1, seq_len, seq_len)
        soft_mask = torch.clamp(soft_mask, 0, 1)  # ensure mask is within [0, 1]

        # Apply the soft_mask without in-place operation
        attn = attn * soft_mask  # element-wise multiplication
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        
        return attn
    

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
        self.adaptive_span = config.adaptive_span
        self.residual_attn = config.residual_attention
        self.residual_attn_mode = config.residual_attention_mode
        self.in_gate = config.in_gate
        self.sparse_topk = config.sparse_topk
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if self.adaptive_span or self.residual_attn:
            self.flash = False
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        if self.rope:
            self.rotary_emb = RoPE(dim=config.n_embd // config.n_head)

        if self.adaptive_span:
            self.adaptive_span = AdaptiveSpan(
                max_span=config.block_size, 
                n_heads=config.n_head,
                adapt_span_loss_coeff=config.adapt_span_loss_coeff,
                ramp_size=config.ramp_size
            )

        if self.in_gate:
            self.in_gate = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.in_gate_activation = F.silu


    def forward(self, x, pos=None, prev_attn=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rope and pos is not None:
            q, k = self.rotary_emb(q, k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            y = self._manual_attention(q, k, v, T, prev_attn)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        if self.in_gate:
            gate = self.in_gate_activation(self.in_gate(x))
            y = y * gate

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, prev_attn
    
    def sparse_topk_attn(self, attn, sparse_topk):
        orig_attn = attn
        mask_value = -torch.finfo(attn.dtype).max
        top_values, _ = attn.topk(sparse_topk, dim = -1)
        sparse_topk_mask = (attn >= top_values[..., -1:]) & (attn > mask_value)
        attn = attn.masked_fill(~sparse_topk_mask, mask_value)
        topk_attn = attn.softmax(dim = -1)
        soft_attn = orig_attn.softmax(dim = -1)
        return topk_attn.detach() + soft_attn - soft_attn.detach()
    
    def _manual_attention(self, q, k, v, T, prev_attn=None):
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        if prev_attn is not None and self.residual_attn:
            if self.residual_attn_mode == 'add':
                att = att + prev_attn
            elif self.residual_attn_mode == 'mean':
                att = (att + prev_attn) / 2
            else:
                raise ValueError(f"Unsupported residual_attention_mode: {self.residual_attn_mode}")
            prev_attn = att
        if self.sparse_topk > 0:
            # Apply sparse top-k attention
            att = self.sparse_topk_attn(
                attn=att,
                sparse_topk=self.sparse_topk,
            )
        else:
            # Apply standard softmax attention
            att = F.softmax(att, dim=-1)
            
        if self.adaptive_span:
            att = self.adaptive_span(att)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
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
    
class GLU(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * int(2 * config.n_embd / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
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
        elif config.mlp == "glu":
            self.mlp = GLU(config)
        else:
            raise ValueError(f"Unsupported MLP type: {config.mlp}")
        self.pre_ln = config.pre_ln

    def forward(self, x, pos=None, prev_attn=None):
        if self.pre_ln:
            # Pre-LN: Apply LayerNorm before sub-layers
            norm_x = self.ln_1(x)
            att_output, _ = self.attn(norm_x, pos)
            x = x + att_output

            # MLP Sub-layer
            norm_x = self.ln_2(x)
            mlp_output = self.mlp(norm_x)
            x = x + mlp_output
        else:
            # Post-LN: Apply LayerNorm after sub-layers
            att_output, prev_attn = self.attn(x, pos, prev_attn)
            x = x + att_output
            x = self.ln_1(x)

            # MLP Sub-layer
            mlp_output = self.mlp(x)
            x = x + mlp_output
            x = self.ln_2(x)
        return x, prev_attn
    
class RoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, q, k):
        t = q.size(-2)
        freqs = torch.einsum("i,j->ij", torch.arange(t, device=q.device).float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        q = self.apply_rotary_pos_emb(q, cos, sin)
        k = self.apply_rotary_pos_emb(k, cos, sin)  # Fix the call for 'k'
        return q, k
    
    def apply_rotary_pos_emb(self, q, cos, sin):
        # Apply rotary position embedding to query and key
        q_cos = q * cos
        q_sin = q * sin
        q_rotated = q_cos + self.rotate_half(q_sin)
        return q_rotated

    def rotate_half(self, x):
        # Helper function to apply rotation
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, n_embd, max_seq_len=512):
        super(PositionalEmbedding, self).__init__()

        self.n_embd = n_embd
        self.max_seq_len = max_seq_len

        # Create a long enough P matrix once.
        pe = torch.zeros(max_seq_len, n_embd)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]

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
    weight_tying: bool = True
    pre_ln: bool = True
    # Modifications to the original GPT-2 config
    # Training
    num_targets: int = 2
    intermediate_loss: bool = False
    max_iters: int = 600000
    # Adaptive span
    adaptive_span: bool = False
    adapt_span_loss_coeff: float = 0.000002
    ramp_size: int = 32
    # Memory tokens
    num_memory_tokens: int = 0
    # Residual attention
    residual_attention: bool = False  
    residual_attention_mode: str = 'add'  
    # Gated input residual in attention
    in_gate: bool = False
    # Sparse top-k attention
    sparse_topk: int = 0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        if config.residual_attention:
            assert not config.pre_ln, "Residual attention is not compatible with Pre-LN"
        self.config = config

        if config.num_memory_tokens > 0:
            self.mem_tokens = nn.Parameter(torch.randn(config.num_memory_tokens, config.n_embd))  # m memory tokens

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
        elif config.pos_emb == 'none':
            pass
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
            elif self.config.pos_emb == 'learned':
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _intermediate_loss(self, x, targets, layer_idx, current_iter):
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
        return sum(intermediate_losses)
    
    def _multi_step_output_loss(self, x, targets):
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
        return logits, sum(losses)
    
    def _initial_pos_emb(self, tok_emb, pos):
        if self.config.pos_emb == 'absolute':
            pos_emb = self.transformer.wpe(pos.size(0))  # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        elif self.config.pos_emb == 'learned':
            pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        elif self.config.pos_emb == 'learned_per_layer':
            x = self.transformer.drop(tok_emb)
        elif self.config.pos_emb is 'none':
            x = self.transformer.drop(tok_emb)
        else:
            raise ValueError(f"Unsupported pos_emb: {self.config.pos_emb}")
        return x

    def forward(self, idx, targets=None, eval_only=False, current_iter=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if self.config.num_memory_tokens > 0:
            # Memory embeddings expanded for the batch
            mem_emb = self.mem_tokens.unsqueeze(0).expand(b, -1, -1)  # (b, m, n_embd)

            # Concatenate memory tokens with input tokens
            tok_emb = torch.cat([mem_emb, tok_emb], dim=1)  # (b, m + t, n_embd)

        # Positional embeddings
        x = self._initial_pos_emb(tok_emb, pos)
        
        total_loss = 0
        prev_attn = None

        for layer_idx, block in enumerate(self.transformer.h):
            if self.config.pos_emb == 'learned_per_layer':
                pos_emb_layer = self.transformer.wpe[layer_idx](pos)
                x = x + pos_emb_layer
            x, prev_attn = block(x, pos if self.config.rope else None, prev_attn)

            # Compute auxiliary predictions from the current layer
            if self.config.intermediate_loss and targets is not None and not eval_only:
                total_loss += self._intermediate_loss(x, targets, layer_idx, current_iter)

        x = self.transformer.ln_f(x)

        if targets is not None and not eval_only:
            logits, loss = self._multi_step_output_loss(x, targets)
            total_loss += loss
        elif targets is not None and eval_only:
            logits = self.lm_heads[0](x)
            total_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_heads[0](x[:, [-1], :]) # note: using list [-1] to preserve the time dim; (b, 1, vocab)
            total_loss = None
        
        if self.config.adaptive_span and not eval_only:
            # go through all CausalSelfAttention blocks and sum up the span loss
            for block in self.transformer.h:
                if hasattr(block.attn, 'adaptive_span'):
                    total_loss += block.attn.adaptive_span.get_loss()

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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

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
