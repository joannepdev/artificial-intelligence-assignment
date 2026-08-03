# Artificial Intelligence Assignment

# Representation Learning with a Variational Auto-Encoder (VAE)

Tabular Q-learning treats every board configuration as an independent state, which limits generalization between similar positions. 
A Variational Autoencoder (VAE) is used to learn a compact latent representation of Tic-Tac-Toe board states. 
The learned latent vectors can capture similarities between different board configurations and provide a more meaningful representation for value estimation.

## Dataset Construction and State Encoding
### Generating a dataset of board states

In this example, a dataset of board states was created. It consists of valid Tic-Tac-Toe board configurations generated through gameplay simulations.
It also collects states from random play episodes.
Since states were generated only after legal moves, invalid configurations are avoided.
An invalid configuration example is the number of impossible differences between X and O marks, where both players have winning lines simultaneously.

