import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def q_learning(q, alpha=0.1, gamma=1.0):
    state = 0  # Start in STATE_A
    left_count = 0
    
    while state != 2:  # Continue until reaching STATE_TERMINAL
        # Epsilon-greedy action selection
        action = (
            np.random.choice(range(len(q[state])))
            if np.random.rand() < 0.1  # EPSILON = 0.1
            else np.random.choice(np.flatnonzero(q[state] == np.max(q[state])))
        )

        # Count "left" actions in STATE_A
        if state == 0 and action == 1:  # ACTION_A_LEFT = 1
            left_count += 1

        # Determine reward
        reward = 0 if state == 0 else np.random.normal(-0.1, 1)  # STATE_A vs STATE_B rewards

        # Determine next state
        next_state = [2, 1][action] if state == 0 else 2  # Transition logic

        # Q-Learning update
        q[state][action] += alpha * (reward + gamma * np.max(q[next_state]) - q[state][action])

        state = next_state

    return left_count


# Double Q-Learning
def double_q_learning(q1, q2, alpha=0.1, gamma=1.0):
    state = 0  # Start in STATE_A
    left_count = 0

    while state != 2:  # Continue until reaching STATE_TERMINAL
        # Epsilon-greedy action selection
        if np.random.rand() < 0.1:  # EPSILON = 0.1
            action = np.random.choice(range(len(q1[state])))
        else:
            # Sum Q-values from q1 and q2 for action selection
            action = np.random.choice(np.flatnonzero(q1[state] + q2[state] == np.max(q1[state] + q2[state])))

        # Count "left" actions in STATE_A
        if state == 0 and action == 1:  # ACTION_A_LEFT = 1
            left_count += 1

        # Determine reward
        reward = 0 if state == 0 else np.random.normal(-0.1, 1)  # STATE_A vs STATE_B rewards

        # Determine next state
        next_state = [2, 1][action] if state == 0 else 2  # Transition logic

        # Randomly choose active and target Q-tables
        active_q, target_q = (q1, q2) if np.random.rand() < 0.5 else (q2, q1)

        # Double Q-Learning update
        best_action = np.random.choice(np.flatnonzero(active_q[next_state] == np.max(active_q[next_state])))
        target = target_q[next_state][best_action]
        active_q[state][action] += alpha * (reward + gamma * target - active_q[state][action])

        state = next_state

    return left_count


# Generate Figure 6.5
def figure_6_5():
    episodes = 300
    runs = 2000

    # Combine data into a dictionary for both Q-Learning and Double Q-Learning
    left_counts = {
        "Q-Learning": np.zeros((runs, episodes)),
        "Double Q-Learning": np.zeros((runs, episodes)),
    }

    for run in tqdm(range(runs)):
        # Initialize independent Q-tables for each run
        q = [np.zeros(2), np.zeros(10), np.zeros(1)]  # INITIAL_Q with ACTIONS_B = range(10)
        q1 = [np.zeros(2), np.zeros(10), np.zeros(1)]  # Independent Q-table for Double Q-Learning
        q2 = [np.zeros(2), np.zeros(10), np.zeros(1)]  # Independent Q-table for Double Q-Learning
        for ep in range(episodes):
            left_counts["Q-Learning"][run, ep] = q_learning(q)
            left_counts["Double Q-Learning"][run, ep] = double_q_learning(q1, q2)

    # Calculate the mean across all runs
    left_counts_mean = {key: values.mean(axis=0) for key, values in left_counts.items()}

    # Plot results
    plt.plot(left_counts_mean["Q-Learning"], label="Q-Learning")
    plt.plot(left_counts_mean["Double Q-Learning"], label="Double Q-Learning")
    plt.axhline(y=0.05, color="gray", linestyle="--", label="Optimal (5%)")  # Optimal percentage = 5%
    plt.xlabel("Episodes")
    plt.ylabel("% Left Actions from A (Optimal = 5%)")  # Updated y-axis label
    plt.legend()
    plt.show()
    plt.close()


if __name__ == '__main__':
    figure_6_5()
