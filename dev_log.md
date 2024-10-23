# Notes
* Code currently does not use entire validation set for validation.
* Performing running validation it seems like the small batch size that the code currently uses is fine - validation loss stabilises quite quickly (within 15 batches) and the first approximation is quite close (within 2 d.p.). 
* Carrying on with the current validation.
* Will perform full validation every 10K.
* Bear in mind that current running validation will probably be an undercount.
* To speed up full validation could use KV cache.
* Proceeding with the small (T12) model size described in the paper.
* We'll keep the position of the layernorm as they are right now (GPT-2 and LLaMA-2 style).


# Things we can modify
Model
- [x] Block size (context size)
- [x] Number of heads
- [x] Head dimension (Embedding dimension)
- [x] Dropout
- [x] Number of layers
- [ ] Layer norm type (RMSNorm/LayerNorm)
- ~~Position of layernorm~~
- [ ] Biases or not
- [ ] Residual scaling
- [ ] Positional embedding type (Absolute/Relative/AliBi/RoPE)
- [ ] Layerwise positional embedding
- [ ] Activation function (GeLU/ReLU,SoLU, SwiGLU)
- [ ] Initialisation scheme (Random/GPT2/...)

Training
- [x] Learning rate
- [ ] Learning rate schedule (Cosine/Adam/IVON)
- [ ] Optimiser (Adam/AdamW/Cosine/IVON)
- [ ] Gradient accumulation steps
- [ ] Batch size
- [ ] Losses (layerwise etc.)
- [x] Training steps
- [ ] Precision

Harder
    * Transformer-XL
    * Compressive transformer
    * Sparse transformer
    * MoE
    * Low Rank
    * Differential transformer
    * DeepSeek transformer
    * Pyramid structure

# Questions
* What is avg. length of a wiki article?
* Do we do one character at a time?
* How many independent characters are there in enwik8? 205
* What is the token distribution? See byte_distribution.png.
* What is the difference between GPT and LLaMA?