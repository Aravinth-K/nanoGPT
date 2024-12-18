# Notes
* Code currently does not use entire validation set for validation.
* Performing running validation it seems like the small batch size that the code currently uses is fine - validation loss stabilises quite quickly (within 15 batches) and the first approximation is quite close (within 2 d.p.). 
* Carrying on with the current validation.
* Will perform full validation every 10K.
* Bear in mind that current running validation will probably be an undercount.
* To speed up full validation could use KV cache.
* Proceeding with the small (T12) model size described in the paper.
* We'll keep the position of the layernorm as they are right now (GPT-2 and LLaMA-2 style).
* One thing I didn't understand in the main paper is that (I think) they predict two tokens ahead using the same context. Therefore we only need to duplicate the head and not do autoregressive prediction.
* Everything currently functioning. Managed to do a training run on a Colab A100 GPU in 10 minutes on 5000 steps that got an estimated validation loss of 1.25.
* Plan now is to fix the number of training steps to 10K.
* To do list is now
    * Fix batching of predictions for multiple targets/intermediate losses to speed up processing.
    * Fix wandb logging.
    * Ensure all checkpoints are saved with appropriate names.
    * Run a 100K step training run. This will serve as the baseline.
    * Make firm decision on use of biases in prediction heads.
    * Create a true evaluation script.
    * One by one go through things we can modify and modify them with all else held fixed.
* Add tests.
* Acknowledge that this is different from Vaswani in that the layernorm is not at the end.
* This is quite boring right now. The main things that we would tune would be:
    * Block size.
    * Number of heads.
    * Head dimension.
    * Layer norm type.
    * Positional embedding type.
    * Biases.
    * Activation function.
    * Optimiser.
* It would be much more interesting to try the discrete diffusion model.
* Currently best performance on 50K steps is ~1.5. Quite high!
* According to Rae et. al. "... reducing the frequency of optimisation updates during training. We find this allows for the best of both worlds — fast initial learning with frequent updates, and better generalisation near the end of training with less frequent updates (e.g. every 4 steps). Reducing the optimisation frequency increases the effective batch size..." 
* Not so interested in adding a memory mechanism. I like how the transformers method of Al-Rfou et. al. didn't incorporate one and it worked well.

# Things we can modify
Model
- [x] Block size (context size)
- [x] Number of heads
- [x] Head dimension (Embedding dimension)
- [x] Dropout
- [x] Number of layers
- [x] Layer norm type (RMSNorm/LayerNorm)
- ~~Position of layernorm~~
- [x] Biases or not
- [x] Activation function (GeLU/ReLU/SoLU, SwiGLU)
- [ ] Residual scaling
- [x] Positional embedding type (singel learned/learned per layer/RoPE)
- [ ] Initialisation scheme (Random/GPT2/...)

Training
- [x] Learning rate
- [ ] Optimiser (Adam/AdamW/Cosine/IVON)
- [x] Gradient accumulation steps
- [x] Batch size
- [x] Multiple predictions
- [x] Intermediate losses
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
* What is avg. length of a wiki article? 8192 bytes according to BP-paper
* Do we do one character at a time?
* How many independent characters are there in enwik8? 205
* What is the token distribution? See byte_distribution.png.
* What is the difference between GPT and LLaMA? LLaMA uses SwiGLU and RMSNorm and RoPE.
* Are we currently doing the "multiple position" training? Yes.
* Do we need multiple prediciton heads for every layer? Maybe.
* Should we add a bias to the prediction heads?
* What is the effect of scaling the loss by 1/ln(2)?
* What is the actual performance of the model that is reported? Is it the loss averaging over predictions at all positions or just the last position?