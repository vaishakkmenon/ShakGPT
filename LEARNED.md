# Learned

A file to keep track of the things I have reviewed/learned while working on this project. I will keep a list of keywords and topics, complex logic, referenced papers, etc.

## Keywords and Topics

Tokens: Characters, subwords or words that are fed into the model. Split is decided based on how tokenization is planned for. 

Embeddings: Mathematical numerical representation of a token. Generally stored in a vector format that is used to perform mathematical operations on.

Attention: An algorithm that uses the current token's embedding, multiplies against three weight matrices (Query, Key, Value), takes the output and compares against the other tokens to identify its meaning, relationship, and context.

Feed-Forward Network: A neural network that is applied to each token's embedding. This is done by moving to a higher dimension, applying an activation function, and then moving back to the original dimension which allows the model to understand factual knowledge and complex transformations.

Normalisation: This is a technique used to deal with the variability of large and small numbers to ensure that the scale of numbers does not create instability in the training.

Positional Encoding: This is a technique used to add information about the position of a token in the sequence. Allows for a token/word to know where it stands chronologically in a sentence and helps the model understand the order of words. The original position encoding uses absolute position which is not ideal for longer sequences. I will be using RoPE which rotates each token's vector by an angle proportional to its position in the sequence, so the model can distinguish tokens that are close together from tokens that are far apart. With RoPE position is relative so words that are further away, are rotated more. Position is supposed to be encoded into the attention vector through rotation due to RoPE.

GQA (Grouped Query Attention): A technique used to attach multiple Q vectors to a single K and V vector. This allows us to reduce the memory used while maintaining quality close to that of Multi-Head Attention. Key factor is that n_heads must be divisible by n_kv_heads with 0 remainder.

SwiGLU (Swish Gated Linear Unit): A feed-forward network, like ReLU, but with a gating mechanism. This ffn projects the input to a higher dimension, runs it through a swish activation which is like ReLU but never truly reaches 0 and then multiplies it with the gating mechanism to determine how much information passes through. Allows for us to learn what information to activate and how much of it to let through.

FlashAttention: An algorithm that computes the attention mechanism in a way that reduces the memory used and increases the speed of computation. Instead of storing the entire matrix in memory, computation is done in small tiles that fit on the GPUs SRAM. Allows us to compute the full matrix without having to store the entire matrix in memory.

Rope Theta: Variable used in RoPE to determine rotation frequencies across dimensions. Higher theta means that the frequency cycling is done slower, allowing the model to capture longer range dependencies.

Tokenizer: The component of the model that is responsible for converting text into tokens and vice versa.

BPE: Byte Pair Encoding is a technique used to create tokens based on the most common pairs of bytes in the training data. This allows for the model to handle special characters.

Byte-Level Encoding: A tokenization strategy that first splits the input text into individual bytes and then uses BPE to merge them into tokens.

Special Tokens: Tokens that are added to the tokenizer to represent special concepts such as the beginning of a sequence, the end of a sequence, and padding. We are using [PAD], [BOS], [EOS] which represent the padding token for attention masks, begin of sequence token for text generation, and end of sequence token for text generation.

Embedding Table: A table that stores the embedding vectors for each token in the vocabulary in which the vectors are learnable through backpropagation. Allows for us to look up the embedding for a token based on its index.

NN.Module: Python class that is the base class for all neural network modules. It provides methods for:
- Initializing the module
- Moving the module to a device
- Moving the module to a data type
- Saving and loading the module
- Paramater tracking when using nn.Embedding or nn.Linear layers.

Weight tying: A technique used to reduce parameters by using the same weight matrix for multiple operations. E.g. InputEmbedding and Output Projection in the Transformer model share the same weight matrix.

Embedding Tensor Shapes: Shape of the data that is being processed by the model. Important for debugging and ensuring that the model is processing the data correctly. Specifically, what I learned about this was:  input is [batch_size, seq_len] of token IDs, output is [batch_size, seq_len, d_model] of vectors.

LayerNorm: Normalisation technique used to scale embeddings along the feature dimension. Specific formula: LayerNorm(x) = (x - mean(x)) / sqrt(variance(x) + eps) + weights.

RMSNorm: Normalisation technique used to scale embeddings along the feature dimension. Specific formula: RMSNorm(x) = x / RMS(X) * weights where RMS(x) = sqrt(mean(x^2)). We chose RMSNorm over LayerNorm because RMSNorm drops the recentering step that LayerNorm performs — subtracting the mean — and only keeps the rescaling. This is simpler and empirically performs just as well.

## Design Choices

config.py-ModelConfig: This is a dataclass that hosts the configuration variables that will be used to determine the specifications of the model. Having a singular location where this information is stored allows for easy access to changes and ensures that all parts of the model are using the same values.

max_seq_len: We are using 2048 for now to keep it manageable and quick to train. It will be increased once we have a stable model.

rope_theta: We are using 10000.0 which matches the RoPE paper value. This value is more appropriate for shorter contexts.

tokenizer: We are using a custom BPE tokenizer trained on the dataset we will be using to train the model. This is because we want to be able to control all aspects of the model to the best of our ability.

## Papers

- Must read: Vaswani et al., 2017 — "Attention Is All You Need"
- Before building attention: Ainslie et al., 2023 — "GQA"
- Before building FFN: Shazeer, 2020 — "GLU Variants Improve Transformers"
- Before building positional encoding: Su et al., 2021 — "RoFormer"
- Extra: Dao et al., 2022 and 2023 — "FlashAttention" and "FlashAttention-2"