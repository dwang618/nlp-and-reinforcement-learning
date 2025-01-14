import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

epsilon = 0.1

def step(state, action):
    # Define environment dimensions
    height, width = 4, 12

    # Define action deltas (dy, dx): Up, Down, Left, Right
    action_deltas = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1)    # Right
    }

    # Get current position and action delta
    y, x = state
    dy, dx = action_deltas[action]

    # Compute next position and clip within boundaries
    new_y = np.clip(y + dy, 0, height - 1)
    new_x = np.clip(x + dx, 0, width - 1)
    next_state = [new_y, new_x]

    # Determine reward and check for cliff condition
    reward = -1
    if is_cliff(next_state, action):
        reward = -100
        next_state = [3, 0]  # Reset to start position

    return next_state, reward

def is_cliff(state, action):
    # Cliff condition: If stepping down on the third row or moving right from start
    y, x = state
    return (y == 3 and 1 <= x <= 10 and action == 1) or (action == 3 and [y, x] == [3, 0])

# Epsilon-greedy action selection
def select_action(state, q_values):
    # Decide whether to explore or exploit
    if should_explore(epsilon):
        return np.random.randint(0, 4)
    else:
        # Get the Q-values for the current state
        y, x = state
        state_values = q_values[y, x, :]

        # Return the action with the highest Q-value (break ties randomly)
        best_action = np.flatnonzero(state_values == np.max(state_values))
        return np.random.choice(best_action)

def should_explore(epsilon):
    # Check if the agent should explore based on the epsilon probability
    return np.random.uniform(0, 1) < epsilon

def sarsa_update(q_values, state, action, next_state, next_action, reward, alpha, gamma):
    """
    Helper function to update the Q-value using the Sarsa update rule.
    """
    y, x = state
    ny, nx = next_state
    td_error = reward + gamma * q_values[ny, nx, next_action] - q_values[y, x, action]
    q_values[y, x, action] += alpha * td_error

def run_sarsa(q_values, alpha=0.5, gamma=1.0, epsilon=0.1):
    """
    Executes a single episode using the Sarsa algorithm.
    """
    current_state = [3, 0]  # Starting position
    target_state = [3, 11]  # Goal position
    total_reward = 0

    # Choose the initial action using the epsilon-greedy strategy
    current_action = select_action(current_state, q_values)

    # Loop until the agent reaches the goal
    while current_state != target_state:
        # Take the action and observe the next state and reward
        next_state, reward = step(current_state, current_action)
        next_action = select_action(next_state, q_values)

        # Accumulate the total reward
        total_reward += reward

        # Update the Q-value using the Sarsa update rule
        sarsa_update(q_values, current_state, current_action, next_state, next_action, reward, alpha, gamma)

        # Move to the next state and action
        current_state = next_state
        current_action = next_action

    return total_reward


# Q-Learning algorithm
def q_learning(q_values, alpha=0.5, gamma=1.0):
    """
    Executes a single episode using the Q-Learning algorithm.
    """
    current_state = [3, 0]  # Start state
    total_reward = 0

    # Loop until the agent reaches the goal state
    while current_state != [3, 11]:
        # Select an action using the epsilon-greedy strategy
        action = select_action(current_state, q_values)

        # Take the action and observe the next state and reward
        next_state, reward = step(current_state, action)

        # Accumulate the total reward
        total_reward += reward

        # Q-Learning update rule:
        # Q(s, a) ← Q(s, a) + α [r + γ * max(Q(s', a')) - Q(s, a)]
        y, x = current_state
        ny, nx = next_state
        best_future_value = np.max(q_values[ny, nx, :])
        td_error = reward + gamma * best_future_value - q_values[y, x, action]
        q_values[y, x, action] += alpha * td_error

        # Move to the next state
        current_state = next_state

    return total_reward


def figure_6_6():
    episodes = 500
    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)

    for r in tqdm(range(1000)):
        q_sarsa = np.zeros((4, 12, 4))
        q_q_learning = np.copy(q_sarsa)
        for i in range(episodes):
            rewards_sarsa[i] += max(run_sarsa(q_sarsa), -100)
            rewards_q_learning[i] += max(q_learning(q_q_learning), -100)

    # Plotting results
    plt.plot(rewards_sarsa/1000, label='Sarsa')
    plt.plot(rewards_q_learning/1000, label='Q-Learning')

    plt.ylim([-100, -10])
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    
    plt.legend()
    plt.show()

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '6_6example.png')
    plt.savefig(file_path)
    plt.close()

if __name__ == '__main__':
    figure_6_6()
