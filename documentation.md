# Artificial Intelligence Assignment

This assignment demonstrates an extension of generalized Tic-Tac-Toe from Dynamic Programming methods
`(Policy Iteration and Value Iteration)` to model-free Reinforcement Learning, specificly Q-Learning method.

Given an agent that learns to play on a nxn board, and a generalized winning condition:

    A player wins by forming a line of length k using their marks in any of the standard Tic-Tac-Toe directions,
    i.e., horizontally, vertically, or diagonally (both main diagonal directions)

the task is to extend the tabular approach by incorporating representation learning through a Variational Auto-Encoder (VAE)
that learns compact latent embeddings of board states, enabling improved generalization to unseen configurations.

# Documentation
## Definition of the Q-table

On my Tic-Tac-Toe environment, the Q-table is initially empty.
It is used as a dictionary, where it will store learning parameters and values needed for learning/training.

Matrix size is determined by the user as well as winning length k (user being prompted to input matrix size and winning length k).

## Board and state description

The board and its state are first initialized and encoded in zeros.
Then, the board is utilized in a value of 0.5, regardless of its size.

## Available and best actions

Available actions bypass each loop on certain instructions set in the for-loop, covering board size.
Actions are chosen through the Q-table.
In case multiple actions share the same value, best actions table is empty.
A random value represents best actions.

## Moves

Moves are made based on:

- action
- player (either Player X or Player Y)
- board

regardless which player's turn it is.
The game randomly decides each player's turn.

## Definition of learning rate α (alpha) and γ (gamma)

### Learning rate α (alpha)
The learning rate α (alpha) represents the learning rate. It means how much the agent changes its current Q-value after each move.
Each value has a different representation on learning and knowledge rates.

- A large α (0.8-1.0) means the agent learns quickly and updates its strategy rapidly.
- A small α (ie 0.1) means the agent updates knowledge slowly and relies more on what has previously learned.

### γ (gamma)
γ (gamma) determines the comparison of important future rewards and immediate rewards.

- High gamma encourages the agent to choose moves that may not give immediate reward. Winning chance is increased.
- Low gamma makes the agent focus mainly on immediate rewards.

## ε-greedy policy exploration

Exploration is necessary after training, so that the agent experiences many different board positions and learns their values.

The ε-greedy policy selects a random number and returns it in available actions (exploration).
Random number value should range between 0 and 1.

## ε-greedy policy exploitation

If epsilon (ε) is too low:

- The agent exploits random Q-values
- Many possible moves are never explored
- The agent makes poor strategy

If epsilon (ε) is too high:

- The agent keeps making random moves
- Slow learning is achieved
- The agent makes unnecessary random decisions, even after learning good moves.

A balance between exploration and exploitation is essential.

## ε-greedy policy decay

An approach to decrease epsilon (ε) during testing is being examined in this question.
In this example, the sequence follows:

- multiplication of epsilon (ε) by 0.995 (decay constant)
- epsilon (ε) value kept in a range above 0.01, which means the minimum exploration probability is 0.01

This approach uses an exponential decay strategy and allows extensive exploration early in training,
mostly exploitation after gaining sufficient experience.

## Random Opponent

A random opponent chooses uniformly among all legal moves.
Random play frequently makes mistakes. The Q-learning agent eventually discovers which moves exploit made mistakes.

After sufficient training, the agent learns strategies that achieve high win rate, 
repeatedly observing successful responses to the opponent's random behavior.

## Minimax Opponent

A minimax opponent attemps to play optimally. Against such an opponent:

- mistakes are rarely available
- winning becomes much more difficult
- the Q-learning agent learns defensive strategies
- the learned policy focuses on forcing draws and avoiding losses rather than exploiting errors

Compared to training against a random opponent, learning is slower because the environment is much more challenging.

## Self-Play

In self play, both players are controlled by the same Q-learning algorithm and share the same Q-table.
Self-play is considered stronger, because:

- both players continuously improve
- difficulty automaticly increases as learning progresses
- overfitting of the agent is not allowed to a single opponent
- different strategies emerge

Finite games such as Tic-Tac-Toe, tabular Q-learning converges to an optimal policy provided that:

- every state-action pair is visited infinitely often
- learning rate satisfies the standard convergence conditions (typically decreasing appropriately over time)
- exploration continues sufficiently
- the environment is stationary during learning

## Unseen States

Some board configurations may never occur either because:

- they are illegal
- the game finishes before reaching them

or:

- exploration never visits them.

Since these states are never updated, their Q-values remain at their initial values (commonly zero).
Therefore, when the agent encounters an unseen state, it has little information for choosing the best action.

## Generalization

Tabular Q-learning stores an independent Q-value for every state-action pair.

- learning one board position does not help similar positions
- each configuration must be learned separately
- the Q-table becomes increasingly large as board size grows
- learning becomes inefficient

As a result, tabular methods generalize poorly.

## Potential Improvements

Several methods can improve generalization:

- Neural networks to approximate Q-values `(Deep Q-Learning)`
- Function approximation using feature representations.
- Representation learning `(e.g., Variational Auto-Encoders)` to encode similar board states into similar latent representations.
- Additional exploration strategies `(ie Upper Confidence Bound (UCB), optimistic initialization)`
- Exploiting board symmetries (rotations and reflections) to reduce the number of unique states.

## Convergence in Tabular Q-Learning

Tabular Q-learning converges to the optimal policy if:

- every state-action pair is visited infinitely often,
- the learning rate satisfies the Robbins–Monro conditions (decreasing over time while remaining sufficient for learning),
- the environment is stationary,
- the discount factor satisfies 0≤γ<1.

A suitable learning rate ensures stable updates, the discount factor balances immediate and future rewards, 
and sufficient exploration guarantees that all actions are eventually evaluated.

## Evaluation Metrics

The learned policy can be evaluated using several performance measures:

- Win rate: Percentage of games won against a random or minimax opponent.
- Draw rate: Percentage of games ending in a draw, particularly against a minimax opponent where draws indicate strong play.
- Loss rate: Percentage of games lost.
- Average number of moves to win: Measures how efficiently the agent wins.
- Average cumulative reward: Mean reward obtained over many evaluation games.
- Training convergence: Monitoring average reward or win rate over episodes to determine whether learning has stabilized.

Using multiple metrics provides a comprehensive assessment of the learned policy's effectiveness, robustness, and learning progress.

