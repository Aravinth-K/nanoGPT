# Notes
* Code currently does not use entire validation set for validation.
* Performing running validation it seems like the small batch size that the code currently uses is fine - validation loss stabilises quite quickly (within 15 batches) and the first approximation is quite close (within 2 d.p.). 
* Carrying on with the current validation.
* Will perform full validation every 10K.
* Bear in mind that current running validation will probably be an undercount.
* To speed up full validation could use KV cache.

# Things we can modify
* Learning rate
* Learning rate schedule (Cosine/Adam/IVON)
* Number of heads
* Head dimension (Embedding dimension)
* Number of layers
* Dropout
* Activation function (GeLU/ReLU,SoLU, SwiGLU)
* Layer norm type (RMSNorm/LayerNorm)
* Initialisation scheme (Random/GPT2/...)
* Optimiser (Adam/AdamW/Cosine/IVON)
* Gradient accumulation steps
* Batch size
* Residual scaling
* Biases or not
* Losses (layerwise etc.)
* Positional embedding type (Absolute/Relative/AliBi/RoPE)
* Learning steps
* Precision
* Harder
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