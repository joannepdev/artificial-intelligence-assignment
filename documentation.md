# Artificial Intelligence Assignment

# Representation Learning with a Variational Auto-Encoder (VAE)

Given two boards a and b, two latent vectors were selected from encoded board states. Intermediate latent vectors were generated using linear interpolation.

(equation)

The interpolated vectors were decoded to examine whether the VAE learned a meaningful continuous latent space.

According to the interpolation results, in the current example:

- Both intermediate tables look valid.
- There are no impossible board states.
- Both boards transited smoothly in terms of interpolation.

#

- In the current example, the number of valid boards is 495 after interpolation.
- 
