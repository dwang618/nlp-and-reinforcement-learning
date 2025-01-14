import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Constants defining the environment
N_STATES = 1000  # Number of states excluding terminal states
START_STATE = 500  # Initial starting state
END_STATES = [0, N_STATES + 1]  # Terminal states
STEP_RANGE = 100  # Maximum stride for an action
ACTIONS = [-1, 1]  # Possible actions: left (-1) and right (+1)
STATES = np.arange(1, N_STATES + 1)

# Helper functions
def compute_true_value():
    """
    Compute the true value function using dynamic programming.
    Iteratively updates the true value estimates until convergence.
    Returns:
        np.ndarray: Array of true values for all states.
    """
    true_value = np.arange(-1001, 1003, 2) / 1001.0
    while True:
        old_value = np.copy(true_value)
        for state in STATES:
            true_value[state] = 0
            for action in ACTIONS:
                for step in range(1, STEP_RANGE + 1):
                    step *= action
                    next_state = state + step
                    next_state = max(min(next_state, N_STATES + 1), 0)

                    # Bellman Equation: V(s) = E[R + γV(s') | s]
                    true_value[state] += 1.0 / (2 * STEP_RANGE) * true_value[next_state]
        error = np.sum(np.abs(old_value - true_value))
        if error < 1e-2:
            break
    true_value[0] = true_value[-1] = 0
    return true_value

def state_estimate(params, group_size, state):
    """
    Get the value of a state based on the current parameters and grouping.
    """
    if state in END_STATES:
        return 0
    group_index = (state - 1) // group_size
    return params[group_index]

def step(state, action):
    """
    State Transition:
        s' = s + stride * action
    """
    step = np.random.randint(1, STEP_RANGE + 1) * action
    state += step
    state = max(min(state, N_STATES + 1), 0)
    reward = -1 if state == 0 else (1 if state == N_STATES + 1 else 0)
    return state, reward

def semi_gradient_temporal_difference(params, group_size, n, alpha):
    """
    Perform the semi-gradient n-step Temporal Difference (TD) algorithm.

    Equation (for n-step TD):
        G_t = R_t+1 + γR_t+2 + ... + γ^(n-1)R_t+n + γ^n * V(s_t+n)
        δ_t = G_t - V(s_t)
        V(s_t) ← V(s_t) + α * δ_t
    """
    state = START_STATE
    states = [state]
    rewards = [0]
    time = 0
    T = float('inf')  # End time for the episode

    while True:
        time += 1
        if time < T:
            action = np.random.choice(ACTIONS)
            next_state, reward = step(state, action)
            states.append(next_state)
            rewards.append(reward)
            if next_state in END_STATES:
                T = time
        
        update_time = time - n
        if update_time >= 0:
            returns = 0.0
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += rewards[t]
            if update_time + n <= T:
                returns += state_estimate(params, group_size, states[update_time + n])
            state_to_update = states[update_time]
            if state_to_update not in END_STATES:
                delta = alpha * (returns - state_estimate(params, group_size, state_to_update))
                if state_to_update in END_STATES:
                    return
                group_index = (state_to_update - 1) // group_size
                params[group_index] += delta
        
        if update_time == T - 1:
            break
        state = next_state

def figure_9_2(true_value):
    """
    Generate the graph for comparing alpha and RMS error for different step sizes.
    Args:
        true_value (np.ndarray): Array of true state values.
    """
    steps = np.power(2, np.arange(0, 10))  # Different step sizes
    alphas = np.arange(0, 1.1, 0.1)  # Different alpha values
    episodes = 10
    runs = 100
    errors = np.zeros((len(steps), len(alphas)))  # Store errors for each (step, alpha)

    for run in tqdm(range(runs), desc="Running simulations"):
        for step_ind, step in enumerate(steps):
            for alpha_ind, alpha in enumerate(alphas):
                group_size = N_STATES // 20
                params = np.zeros(20)

                for ep in range(episodes):
                    semi_gradient_temporal_difference(params, group_size, step, alpha)
                    state_value = np.asarray([state_estimate(params, group_size, i) for i in STATES])

                    # RMS Error Equation: RMS = √(Σ(V_true(s) - V_approx(s))^2 / N)
                    errors[step_ind, alpha_ind] += np.sqrt(np.mean(np.power(state_value - true_value[1:-1], 2)))
    
    # Average the errors over episodes and runs
    errors /= (episodes * runs)

    # Plot the results
    for i in range(len(steps)):
        plt.plot(alphas, errors[i, :], label=f'n = {steps[i]}')
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend(fontsize=8)
    plt.title("RMS Error vs Alpha for Different Step Sizes")
    plt.show()

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '9.2ex.png')
    plt.savefig(file_path)

if __name__ == '__main__':
    # Compute the true values using dynamic programming
    true_value = compute_true_value()

    # Generate the plot for figure 9.2
    figure_9_2(true_value)
