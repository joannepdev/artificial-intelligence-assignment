# Artificial Intelligence Assignment

## Convergence in Tabular Q-Learning

In this question, it is important the Bellman equation be used. This means that every state-action pair is visited infinitely often, 
as well as the learning rate meets appropriate conditions (decreasing over time while sufficient for learning) 
in a stationery environment.
Moreover, the discount factor should have to satisfy the condition below.

    0≤γ<1

To sum up, a suitable learning rate ensures stable updates, controlling how much new information overrides old Q-values.
The discount factor balances immediate and future rewards, whereas sufficient exploration guarantees all actions be evaluated.

## Evaluation Metrics

The learned policy can be evaluated using several performance measures:

- Win rate: Percentage of games won against a random or minimax opponent.
- Draw rate: Percentage of games ending in a draw, particularly against a minimax opponent where draws indicate strong play.
- Loss rate: Percentage of games lost.
- Average number of moves to win: Measures how efficiently the agent wins.
- Average cumulative reward: Mean reward obtained over many evaluation games.
- Training convergence: Monitoring average reward or win rate over episodes to determine whether learning has stabilized.

Using multiple metrics provides a comprehensive assessment of the learned policy's effectiveness, robustness, and learning progress.

