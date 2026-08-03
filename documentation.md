# Artificial Intelligence Assignment

# Representation Learning with a Variational Auto-Encoder (VAE)

Tabular Q-learning treats every board configuration as an independent state, which limits generalization between similar positions. 
A Variational Autoencoder (VAE) is used to learn a compact latent representation of Tic-Tac-Toe board states. 
The learned latent vectors can capture similarities between different board configurations and provide a more meaningful representation for value estimation.

## Dataset Construction and State Encoding
### Generating a dataset of board states

In this example, a dataset of board states was created. It consists of valid Tic-Tac-Toe board configurations generated through gameplay simulations.
It also collects states from 5000 random play episodes.
Since states were generated only after legal moves, invalid configurations are avoided.
An invalid configuration example is the number of impossible differences between X and O marks, where both players have winning lines simultaneously.

### Numerical Encoding

In my implemented code for the Tic-Tac-Toe game simulation application, the board states were encoded using a single-channel numerical representation.
Each cell was represented by a single value where:
- 1 denotes player X
- −1 denotes player O,
- 0 denotes an empty cell.

Each n×n board was then flattened into a one-dimensional vector using NumPy's flatten() function before being used as input to the VAE. 
This representation preserves the complete board information while providing a compact numerical input suitable for neural network training.

Finally, the encoded states were converted into a table, as input to the subsequent train-test split of the dataset.

### Train/Validation-test split
