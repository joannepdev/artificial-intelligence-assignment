import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.decomposition import PCA

# prompt user to enter board size n and winning length k
n = int(input("Enter board size n: "))
k = int(input("Enter winning length k: "))

# define players
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

# set and create dictionary
Q = {}

# set learning parameters
alpha = 0.1
gamma = 0.95
epsilon = 0.1

# create the board
def create_board():
    board = np.zeros((n, n), dtype=int)
    return board

# test board creation

board = create_board()
print(board)

# encode the state
def encode_state(board):
    return tuple(board.flatten())

# test the state
state = encode_state(board)
print(state)

# utilize state in the Q-table
Q[(state, (0,0))] = 0.5

# find available actions
def get_available_actions(board):
    actions = []

    for row in range(n):
        for col in range(n):
            if board[row][col] == EMPTY:
                actions.append((row, col))

    return actions

# test available actions
actions = get_available_actions(board)
print(actions)

# set ε-greedy policy
def choose_action(state, available_actions):
    
    # Exploration
    if random.random() < epsilon:
        return random.choice(available_actions)

    # Exploitation
    else:
        q_values = []

        for action in available_actions:
            q_values.append(Q.get((state, action), 0.0))

        max_q = max(q_values)

        # In case multiple actions have same value
        best_actions = []

        for action, value in zip(available_actions, q_values):
            if value == max_q:
                best_actions.append(action)

        return random.choice(best_actions)

# set ε-greedy decay
def decay_epsilon():
    global epsilon

    epsilon = max(
        0.01,
        epsilon * 0.995
    )

# make a move on the board
def make_move(board, action, player):
    row, col = action

    if board[row, col] == EMPTY:
        board[row, col] = player
        return True

    return False

# check winner
def check_winner(board, player):

    board_size = board.shape[0]

    directions = [
        (0,1),    # horizontal
        (1,0),    # vertical
        (1,1),    # diagonal \
        (1,-1)    # diagonal /
    ]

    for row in range(board_size):
        for col in range(board_size):

            if board[row, col] != player:
                continue

            for dr, dc in directions:

                count = 0

                for step in range(k):

                    r = row + step * dr
                    c = col + step * dc

                    if (
                        0 <= r < board_size and
                        0 <= c < board_size and
                        board[r,c] == player
                    ):
                        count += 1
                    else:
                        break

                if count == k:
                    return True

    return False

# check draw
def check_draw(board):
    return not np.any(board == EMPTY)

# get reward
def get_reward(board, player):

    # Player wins
    if check_winner(board, player):
        return 1

    # Opponent wins
    if check_winner(board, -player):
        return -1

    # Draw
    if check_draw(board):
        return 0

    # Game continues
    return 0

# calculate reward
def calculate_reward(board):

    if check_winner(board, PLAYER_X):
        return 1

    elif check_winner(board, PLAYER_O):
        return -1

    elif check_draw(board):
        return 0

    else:
        return 0

# generate random opponent
def random_opponent(board):

    available_actions = get_available_actions(board)

    return random.choice(available_actions)

# test random opponent
board = create_board()
action = random_opponent(board)
print(action)

# update Q-table
def update_q_table(state, action, reward, next_state, next_actions):
    current_q = Q.get((state, action), 0.0)

    if next_actions:
        max_future_q = max(
            Q.get((next_state, a), 0.0)
            for a in next_actions
        )
    else:
        max_future_q = 0.0

    Q[(state, action)] = current_q + alpha * (
        reward + gamma * max_future_q - current_q
    )

# play episode
def play_episode():

    board = create_board()
    game_over = False

    while not game_over:

        # Current state
        state = encode_state(board)

        # Agent chooses an action
        available_actions = get_available_actions(board)
        action = choose_action(state, available_actions)

        # Agent makes its move
        make_move(board, action, PLAYER_X)

        # Check if the agent won
        if check_winner(board, PLAYER_X):
            reward = 1
            next_state = encode_state(board)

            update_q_table(state, action, reward, next_state, [])

            return reward

        # Check for draw
        if check_draw(board):
            reward = 0
            next_state = encode_state(board)

            update_q_table(state, action, reward, next_state, [])

            return reward

        # Opponent plays
        opponent_action = random_opponent(board)
        make_move(board, opponent_action, PLAYER_O)

        # Opponent wins
        if check_winner(board, PLAYER_O):
            reward = -1
            next_state = encode_state(board)

            update_q_table(state, action, reward, next_state, [])

            return reward

        # Draw after opponent move
        if check_draw(board):
            reward = 0
            next_state = encode_state(board)

            update_q_table(state, action, reward, next_state, [])

            return reward

        # Continue game
        next_state = encode_state(board)
        next_actions = get_available_actions(board)

        reward = 0

        update_q_table(
            state,
            action,
            reward,
            next_state,
            next_actions
        )

# test 10000 episodes with minimax
episodes = 10000
reward_history = []

for episode in range(episodes):
    reward = play_episode()
    reward_history.append(reward)
    decay_epsilon()

# inspect learned state-action pairs
print("Number of learned state-action pairs:", len(Q))

# inspect the first 10 learned state-action pairs
count = 0

for key, value in Q.items():
    print(key, ":", value)
    count += 1

    if count == 10:
        break

# self-play episode
def play_self_play_episode():
    board = create_board()

    while True:

        # -----------------
        # Player X turn
        # -----------------

        state = encode_state(board)

        actions = get_available_actions(board)

        action = choose_action(state, actions)

        make_move(board, action, PLAYER_X)

        next_state = encode_state(board)

        # Check X win
        if check_winner(board, PLAYER_X):
            update_q_table(
                state,
                action,
                1,
                next_state,
                []
            )

            return 1


        # Check draw
        if check_draw(board):
            update_q_table(
                state,
                action,
                0,
                next_state,
                []
            )

            return 0


        # -----------------
        # Player O turn
        # -----------------

        state = encode_state(board)

        actions = get_available_actions(board)

        action = choose_action(state, actions)

        make_move(board, action, PLAYER_O)

        next_state = encode_state(board)


        # Check O win
        if check_winner(board, PLAYER_O):
            update_q_table(
                state,
                action,
                1,
                next_state,
                []
            )

            return -1


        # Check draw
        if check_draw(board):
            update_q_table(
                state,
                action,
                0,
                next_state,
                []
            )

            return 0

# calculate rate percentage
def calculate_percentage(value, total):
    return (value / total) * 100

# evaluate the agent
def evaluate_agent(num_games):

    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):

        reward = play_episode()

        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1

    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")

    print(f"Win Rate: {calculate_percentage(wins, num_games):.2f}%")
    print(f"Loss Rate: {calculate_percentage(losses, num_games):.2f}%")
    print(f"Draw Rate: {calculate_percentage(draws, num_games):.2f}%")

# evaluate agent over 1000 games
evaluate_agent(1000)

# create a moving average
window = 100
moving_average = []

for i in range(len(reward_history)):

    start = max(0, i - window + 1)

    average = sum(reward_history[start:i+1]) / (i - start + 1)

    moving_average.append(average)

# visualize moving average
plt.figure(figsize=(8,5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Moving Average Reward")

plt.show()

# combine Q-tables
def combine_q_tables(Q1, Q2, Q3):

    combined = {}

    all_keys = set(Q1.keys()) | set(Q2.keys()) | set(Q3.keys())

    for key in all_keys:

        value1 = Q1.get(key, 0)
        value2 = Q2.get(key, 0)
        value3 = Q3.get(key, 0)

        combined[key] = (value1 + value2 + value3) / 3

    return combined

# Variational Auto-Encoder

# create dataset states
dataset_states = []

# create a game states collector
def collect_game_states():

    board = create_board()

    states = []

    while True:

        # Save current board
        states.append(board.copy())


        # Player X move
        available_actions = get_available_actions(board)

        action = random.choice(available_actions)

        make_move(board, action, PLAYER_X)


        # Check X win
        if check_winner(board, PLAYER_X):
            states.append(board.copy())
            return states


        # Check draw
        if check_draw(board):
            states.append(board.copy())
            return states



        # Player O move
        available_actions = get_available_actions(board)

        action = random.choice(available_actions)

        make_move(board, action, PLAYER_O)


        # Check O win
        if check_winner(board, PLAYER_O):
            states.append(board.copy())
            return states


        # Check draw
        if check_draw(board):
            states.append(board.copy())
            return states

for episode in range(5000):

    game_states = collect_game_states()
    dataset_states.extend(game_states)

# convert boards into vectors
encoded_states = []

for board in dataset_states:
    encoded_states.append(board.flatten())

# convert to numpy array
X = np.array(encoded_states)

# print array shape
print(X.shape)

# train-test split
X_train, X_temp = train_test_split(
    X,
    test_size=0.3,
    random_state=42
)

X_val, X_test = train_test_split(
    X_temp,
    test_size=0.5,
    random_state=42
)

# get sizes
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# create variational auto-encoder
X_train = X_train.astype("float32")
X_val = X_val.astype("float32")
X_test = X_test.astype("float32")

input_dim = 25
latent_dim = 16

# build encoder
encoder_input = layers.Input(shape=(input_dim,))
x = layers.Dense(64, activation="relu")(encoder_input)
mu = layers.Dense(latent_dim)(x)
log_var = layers.Dense(latent_dim)(x)


# reparametrization
def sampling(args):
    mu, log_var = args

    epsilon = tf.random.normal(
        shape=tf.shape(mu)
    )

    sigma = tf.exp(0.5 * log_var)

    return mu + sigma * epsilon


z = layers.Lambda(sampling)([mu, log_var])

# apply reparametrization
encoder = Model(
    encoder_input,
    [mu, log_var, z]
)

# build decoder
decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(decoder_input)
x = layers.Dense(64, activation="relu")(x)

decoder_output = layers.Dense(
    input_dim,
    activation="tanh"
)(x)

decoder = Model(
    decoder_input,
    decoder_output
)

# recreate the VAE

# define VAE class
class VAE(Model):

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def call(self, inputs):

        mu, log_var, z = self.encoder(inputs)

        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.square(inputs - reconstruction)
        )

        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_var
            - tf.square(mu)
            - tf.exp(log_var)
        )

        self.add_loss(
            reconstruction_loss + 0.001 * kl_loss
        )

        return reconstruction


vae = VAE(
    encoder,
    decoder
)

# compile variational auto-encoder
vae.compile(
    optimizer=tf.keras.optimizers.Adam()
)

# train
history = vae.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=50,
    batch_size=128
)

# reconstruct test boards
X_reconstructed = vae.predict(X_test)
print(X_reconstructed.shape)

# convert reconstructed values back to Tic-Tac-Toe symbols
X_reconstructed_discrete = np.where(
    X_reconstructed > 0.5,
    1,
    np.where(
        X_reconstructed < -0.5,
        -1,
        0
    )
)

# calculate accuracy per cell
accuracy = np.mean(
    X_reconstructed_discrete == X_test
)

print(
    f"Reconstruction accuracy: {accuracy*100:.2f}%"
)

# latent space visualization
mu_values, log_var_values, z_values = encoder.predict(X_test)
print(mu_values.shape)

# calculate occupied cells
occupied_cells = np.sum(
    X_test != 0,
    axis=1
)

# print occupied cells
print(occupied_cells[:10])

# apply PCA
pca = PCA(
    n_components=2
)

# transform and check latent space
latent_2d = pca.fit_transform(mu_values)
print(latent_2d.shape)

# plot latent space
plt.figure(figsize=(8,6))

plt.scatter(
    latent_2d[:,0],
    latent_2d[:,1],
    c=occupied_cells,
    alpha=0.5
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("VAE Latent Space of Tic-Tac-Toe Boards")

plt.colorbar(
    label="Occupied Cells"
)

plt.show()

board_a_index = 0
board_b_index = 100

z_a = mu_values[board_a_index]
z_b = mu_values[board_b_index]

# create interpolation points
interpolation_steps = np.linspace(0, 1, 5)
interpolated_z = []

for t in interpolation_steps:
    z_t = (1-t)*z_a + t*z_b
    interpolated_z.append(z_t)

interpolated_z = np.array(interpolated_z)

# check interpolation shapes
print(interpolated_z.shape)

# decode latent vectors
decoded_boards = decoder.predict(interpolated_z)

# check decoded vectors
print(decoded_boards.shape)

# convert outputs back to board symbols
decoded_boards_discrete = np.where(
    decoded_boards > 0.5,
    1,
    np.where(
        decoded_boards < -0.5,
        -1,
        0
    )
)

# display interpolation sequence
for i, board in enumerate(decoded_boards_discrete):

    print(f"Step {i}")
    print(board.reshape(5,5))
    print()

# generate random latent vectors
num_samples = 10

z_samples = np.random.normal(
    size=(num_samples, latent_dim)
)

print(z_samples.shape)

# decode latent vectors
sampled_boards = decoder.predict(z_samples)

print(sampled_boards.shape)

# convert decoder output to Tic-Tac-Toe symbols
sampled_boards_discrete = np.where(
    sampled_boards > 0.5,
    1,
    np.where(
        sampled_boards < -0.5,
        -1,
        0
    )
)

# display generated boards
for i, board in enumerate(sampled_boards_discrete):

    print(f"Sample {i}")
    print(board.reshape(5,5))
    print()

# sample new latent points
num_generated_samples = 1000

z_generated = np.random.normal(
    size=(num_generated_samples, latent_dim)
)

print(z_generated.shape)

# decode to board states
generated_boards = decoder.predict(z_generated)
print(generated_boards.shape)

# convert to board states
generated_boards_discrete = np.where(
    generated_boards > 0.5,
    1,
    np.where(
        generated_boards < -0.5,
        -1,
        0
    )
)

# extract visited states
visited_states = set(Q.keys())

# print visited states
print("Visited states:", len(visited_states))

# convert generated states into comparable states
generated_states = []

for board in generated_boards_discrete:

    state = tuple(board)
    generated_states.append(state)

# find unseen states
unseen_states = []

for state in generated_states:
    if state not in visited_states:
        unseen_states.append(state)


print("Generated states:", len(generated_states))
print("Unseen states:", len(unseen_states))

# check if board is valid
def is_valid_board(board):

    x_count = np.sum(board == PLAYER_X)
    o_count = np.sum(board == PLAYER_O)

    if abs(x_count - o_count) > 1:
        return False

    return True

# filter generated boards
valid_boards = []

for board in generated_boards_discrete:

    board = board.reshape(n, n)

    if is_valid_board(board):
        valid_boards.append(board)

print("Generated boards:", len(generated_boards_discrete))
print("Valid boards:", len(valid_boards))

# compute acceptance rate
acceptance_rate = len(valid_boards) / len(generated_boards_discrete)
print(f"Acceptance rate: {acceptance_rate*100:.2f}%")

# filter unseen boards
unseen_boards = []

for board in valid_boards:

    state = tuple(board.flatten())

    if state not in visited_states:
        unseen_boards.append(board)

print("Unseen boards:", len(unseen_boards))

# compare both agents
def baseline_action(board):

    state = tuple(board.flatten())

    if state in Q:

        available_actions = get_available_actions(board)

        if not available_actions:
            return None

        q_values = Q[state]

        best_action = max(
            available_actions,
            key=lambda a: q_values[a]
        )

        return best_action

    else:
        available_actions = get_available_actions(board)

        if not available_actions:
            return None

        return random.choice(available_actions)

# compare VAE agent with baseline agent

def vae_action(board):

    # encode current board
    z_board = encoder.predict(
        board.reshape(1, -1),
        verbose=0
    )[0]


    # calculate distances
    distances = np.linalg.norm(
        latent_states - z_board,
        axis=1
    )


    # find 10 nearest latent states
    nearest_indices = np.argpartition(
        distances,
        10
    )[:10]


    estimated_q = {}

    sigma = max(
        np.median(distances[nearest_indices]),
        1e-8
    )


    for action in get_available_actions(board):

        numerator = 0
        denominator = 0

        for i in nearest_indices:

            state = tuple(latent_boards[i])

            if (state, action) in Q:

                distance = distances[i]

                weight = np.exp(
                    -(distance**2) / (sigma**2)
                )

                numerator += weight * Q[(state, action)]
                denominator += weight


        if denominator > 0:
            estimated_q[action] = numerator / denominator


    if estimated_q:

        return max(
            estimated_q,
            key=estimated_q.get
        )


    return random.choice(
        get_available_actions(board)
    )

# check if an action is safe

def is_safe_action(board, action, agent_player, opponent_player):

    # copy board so the original is not modified
    test_board = board.copy()

    # simulate agent's move
    make_move(
        test_board,
        action,
        agent_player
    )

    # check opponent's possible moves
    opponent_actions = get_available_actions(test_board)

    for opp_action in opponent_actions:

        temp_board = test_board.copy()

        make_move(
            temp_board,
            opp_action,
            opponent_player
        )

        # opponent can win immediately
        if check_winner(
            temp_board,
            opponent_player
        ):
            return False

    return True

# evaluate safe actions for both agents

def evaluate_safe_actions(unseen_boards):

    baseline_safe = 0
    vae_safe = 0
    evaluated_boards = 0

    for board in unseen_boards:

        # get actions from both agents
        baseline_move = baseline_action(board)
        vae_move = vae_action(board)

        # skip if an agent cannot move
        if baseline_move is None or vae_move is None:
            continue


        # check baseline safety
        if is_safe_action(
            board,
            baseline_move,
            PLAYER_O,
            PLAYER_X
        ):
            baseline_safe += 1


        # check VAE safety
        if is_safe_action(
            board,
            vae_move,
            PLAYER_X,
            PLAYER_O
        ):
            vae_safe += 1


        evaluated_boards += 1


    baseline_rate = baseline_safe / evaluated_boards * 100
    vae_rate = vae_safe / evaluated_boards * 100

    return baseline_rate, vae_rate