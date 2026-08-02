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

The learned policy can be evaluated using several performance measures.

Win rate is the percentage of games won against a random or minimax opponent.

Draw rate is the percentage of games ending in a draw, particularly against a minimax opponent where draws indicate strong play.

Loss rate is the percentage of games lost.

The average number of winning moves, measures how efficiently the agent wins.

The average cumulative reward stands for the mean reward obtained over many evaluation games.

The training convergence is to monitor average reward or win rate over episodes to determine whether learning has stabilized.

Using multiple metrics provides a comprehensive assessment of the learned policy's effectiveness, robustness, and learning progress.

## Learning Curves

The performance of the Q-learning agent can be monitored by plotting learning curves over the training episodes. Common metrics include the running average of the reward per episode, the moving average of the win percentage, and the win, draw, and loss rates against different opponents. These plots provide insight into the learning progress and help determine whether the agent has converged to a stable policy.

## Combining Knowledge from Multiple Q-Tables
### Different Training Regimes

Different Q-tables may assign different values to the same state-action pair because they are trained under different opponent strategies. A policy learned against a random opponent focuses on exploiting mistakes, while a policy learned against a minimax opponent emphasizes defensive play and avoiding losses. Self-play continuously adapts as both players improve, often resulting in a more balanced strategy.

### Direct Combination

One possible approach is to combine the Q-values by averaging the corresponding entries.

Alternatively, the maximum Q-value for each state-action pair could be selected. However, averaging may weaken strong strategies learned in specific environments, while selecting the maximum may introduce inconsistencies if different Q-tables recommend conflicting actions.

### Further Fine-Tuning

After merging the Q-tables, additional training can be performed using a mixture of opponent types. For example, the agent may alternate between random, self-play, and minimax opponents during training. This allows the merged Q-table to adapt to different playing styles and gradually resolve conflicting Q-values.

## Additional Considerations
### Hyperparameter Sensitivity

The learning rate (α) determines how quickly the Q-values are updated. A large value may lead to unstable learning, whereas a small value slows convergence. The discount factor (γ) controls the importance of future rewards; larger values encourage long-term planning, while smaller values emphasize immediate rewards. The exploration rate (ϵ) balances exploration and exploitation. High values promote exploration, whereas low values encourage the agent to exploit its current knowledge.

### Partially Observable or Stochastic Extensions

If some game information becomes hidden or if actions or observations become noisy, the current state representation may no longer be sufficient. In such cases, the agent could use an enhanced state representation that incorporates previous observations or employ function approximation methods, such as neural networks, to better handle uncertainty and generalize across similar states.

### Time Constraints and Pruning

As the board size increases, the number of possible states grows exponentially, making tabular Q-learning and minimax computationally expensive. To address this issue, the minimax search depth can be limited, heuristic evaluation functions can be used to estimate non-terminal positions, alpha-beta pruning can reduce the number of explored nodes, and approximate Q-functions based on neural networks can replace the tabular representation for larger state spaces.
