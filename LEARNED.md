# Learned

A file to keep track of the things I have reviewed/learned while working on this project. I will keep a list of keywords and topics, complex logic, referenced papers, etc.

## Keywords and Topics

Tokens: Characters, subwords or words that are fed into the model. Split is decided based on how tokenization is planned for. 

Embeddings: Mathematical numerical representation of a token. Generally stored in a vector format that is used to perform mathematical operations on.

Attention: An algorithm that uses the current token's embedding, multiplies against three weight matrices (Query, Key, Value), takes the output and compares against the other tokens to identify its meaning, relationship, and context.

Feed-Forward Network: A neural network that is applied to each token's embedding. This is done by moving to a higher dimension, applying an activation function, and then moving back to the original dimension which allows the model to understand factual knowledge and complex transformations.

Normalisation: This is a technique used to deal with the variability of large and small numbers to ensure that the scale of numbers does not create instability in the training.

Positional Encoding: This is a technique used to add information about the position of a token in the sequence. Allows for a token/word to know where it stands chronologically in a sentence and helps the model understand the order of words. The original position encoding uses absolute position which is not ideal for longer sequences. I will be using RoPE which rotates each token's vector by an angle proportional to its position in the sequence, so the model can distinguish tokens that are close together from tokens that are far apart. With RoPE position is relative so words that are further away, are rotated more.

GQA (Grouped Query Attention): A technique used to attach multiple Q vectors to a single K and V vector. This allows us to reduce the memory used while maintaining quality close to that of Multi-Head Attention. Key factor is that n_heads must be divisible by n_kv_heads with 0 remainder.

SwiGLU (Swish Gated Linear Unit): A feed-forward network, like ReLU, but with a gating mechanism. This ffn projects the input to a higher dimension, runs it through a swish activation which is like ReLU but never truly reaches 0 and then multiplies it with the gating mechanism to determine how much information passes through. Allows for us to learn what information to activate and how much of it to let through.

FlashAttention: An algorithm that computes the attention mechanism in a way that reduces the memory used and increases the speed of computation. Instead of storing the entire matrix in memory, computation is done in small tiles that fit on the GPUs SRAM. Allows us to compute the full matrix without having to store the entire matrix in memory.

Rope Theta: Variable used in RoPE to determine rotation frequencies across dimensions. Higher theta means that the frequency cycling is done slower, allowing the model to capture longer range dependencies.

## Design Choices

config.py-ModelConfig: This is a dataclass that hosts the configuration variables that will be used to determine the specifications of the model. Having a singular location where this information is stored allows for easy access to changes and ensures that all parts of the model are using the same values.

max_seq_len: We are using 2048 for now to keep it manageable and quick to train. It will be increased once we have a stable model.

rope_theta: We are using 10000.0 which matches the RoPE paper value. This value is more appropriate for shorter contexts.

## Papers

- Must read: Vaswani et al., 2017 — "Attention Is All You Need"
- Before building attention: Ainslie et al., 2023 — "GQA"
- Before building FFN: Shazeer, 2020 — "GLU Variants Improve Transformers"
- Before building positional encoding: Su et al., 2021 — "RoFormer"
- Extra: Dao et al., 2022 and 2023 — "FlashAttention" and "FlashAttention-2"