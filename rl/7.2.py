import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Function to implement the n-step Temporal Difference (TD) learning algorithm
# @value: array of state values to be updated
# @n: number of steps for the TD update
# @alpha: learning rate or step size
# @gamma: discount factor is set to 1 since the problem does not involve long-term rewards
def temporal_difference(value, n, alpha):
    start_state = 10  # Middle state to start the episode
    end_states = [0, 20]  # Terminal states where the episode ends

    state = start_state  # Initialize current state
    states = [state]  # Track states visited in this episode
    rewards = [0]  # Track rewards received, initialized with zero for consistency

    time = 0  # Current time step
    T = float('inf')  # End of the episode (initialized to infinity)

    # Run the episode until it terminates
    while True:
        time += 1  # Increment time step

        if time < T:  # If the episode is ongoing
            # Simulate random action: move left or right with equal probability
            next_state = state + 1 if np.random.rand() < 0.5 else state - 1

            # Determine the reward for transitioning to the next state
            reward = -1 if next_state == 0 else (1 if next_state == 20 else 0)

            # Append the new state and reward to their respective lists
            states.append(next_state)
            rewards.append(reward)

            # If the new state is terminal, update the episode's end time
            if next_state in end_states:
                T = time

        # Calculate the time step to update
        update_time = time - n

        if update_time >= 0:  # If we are ready to update a state value
            # Compute the return (G) for the n-step transition
            returns = sum(rewards[t] for t in range(update_time + 1, min(T, update_time + n) + 1))

            # Add the bootstrapped value if the n-step transition extends beyond the episode
            if update_time + n <= T:
                returns += value[states[update_time + n]]

            # Update the value of the state at `update_time` if it is not terminal
            state_to_update = states[update_time]
            if state_to_update not in end_states:
                value[state_to_update] += alpha * (returns - value[state_to_update])

        # Exit the loop if the last state to update has been processed
        if update_time == T - 1:
            break

        # Move to the next state
        state = states[-1]

# Function to generate Figure 7.2: Performance of n-step TD as a function of alpha
def figure7_2():
    # Number of non-terminal states (states between terminal states)
    n_states = 19

    # True values for each state computed from the Bellman equation
    true_value = np.linspace(-1, 1, n_states + 2)  # Linear interpolation between terminal values
    true_value[0] = true_value[-1] = 0  # Terminal states have a value of 0

    # Range of n-step values to test (powers of 2 from 2^0 to 2^9)
    steps = np.power(2, np.arange(0, 10))

    # Range of alpha (step sizes) to test, from 0 to 1.0 in increments of 0.1
    alphas = np.arange(0, 1.1, 0.1)

    # Initialize an array to track RMS errors for each combination of n and alpha
    errors = np.zeros((len(steps), len(alphas)))

    # Perform multiple independent runs to average the results
    for _ in tqdm(range(100)):  # tqdm provides a progress bar
        for i, n in enumerate(steps):  # Iterate over all n-step values
            for j, alpha in enumerate(alphas):  # Iterate over all alpha values
                value = np.zeros(n_states + 2)  # Initialize state values to zero
                for _ in range(10):  # Simulate each episode
                    # Update state values using the n-step TD method
                    temporal_difference(value, n, alpha)

                    # Accumulate the RMS error for the current n and alpha
                    errors[i, j] += np.sqrt(np.sum((value - true_value) ** 2) / n_states)

    # Average the accumulated RMS errors over all episodes and runs
    errors /= 10 * 100

    # Plot the results for each n-step value as a function of alpha
    for i, n in enumerate(steps):
        plt.plot(alphas, errors[i], label=f'n = {n}')  # Plot RMS error vs alpha

    # Add labels and legend to the plot
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])  # Set y-axis limits for better visibility
    plt.legend()
    plt.show()
    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '7.2ex.png')
    plt.savefig(file_path)

# Entry point of the script
if __name__ == '__main__':
    figure7_2()
