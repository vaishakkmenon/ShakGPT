# Learned

A file to keep track of the things I have reviewed/learned while working on this project. I will keep a list of keywords and topics, complex logic, referenced papers, etc.

## Keywords and Topics

Tokens: Characters, subwords or words that are fed into the model. Split is decided based on how tokenization is planned for. 

Embeddings: Mathematical numerical representation of a token. Generally stored in a vector format that is used to perform mathematical operations on.

Attention: An algorithm that uses the current token's embedding, multiplies against three weight matrices (Query, Key, Value), takes the output and compares against the other tokens to identify its meaning, relationship, and context.

Feed-Forward Network: A neural network that is applied to each token's embedding. This is done by moving to a higher dimension, applying an activation function, and then moving back to the original dimension which allows the model to understand factual knowledge and complex transformations.

Normalisation: This is a technique used to deal with the variability of large and small numbers to ensure that the scale of numbers does not create instability in the training.

Positional Encoding: This is a technique used to add information about the position of a token in the sequence. Allows for a token/word to know where it stands chronologically in a sentence and helps the model understand the order of words. I will be using RoPE which rotates each token's vector by an angle proportional to its position in the sequence, so the model can distinguish tokens that are close together from tokens that are far apart.

## Design Choices

config.py-ModelConfig: This is a dataclass that hosts the configuration variables that will be used to determine the specifications of the model. Having a singular location where this information is stored allows for easy access to changes and ensures that all parts of the model are using the same values.