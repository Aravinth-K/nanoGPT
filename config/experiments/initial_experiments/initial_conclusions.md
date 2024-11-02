1. RMSNorm beats LayerNorm.
2. Bias is helpful.
3. Weight tying helpful.
4. 8 heads is better than 4 heads.
5. Larger batch size is better.
6. Smaller learning rate is better.
7. Intermediate loss is better than not using it.
8. Higher number of targets is better.

Things I am not convinced about.
1. Longer warmup is better.
2. Relu is better than other activations.
3. 0.2 dropout is better than 0.1
4. Fewer layers is better.
5. Plateauing the learning rate early is better than not.