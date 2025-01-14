import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
import os  # Import os module to handle directory paths
matplotlib.use('Agg')

grid_size = 4
prob_action = 0.25
gamma = 1.0

# Policy: a random policy initially (equal probability for all actions)
policy = np.ones((grid_size, grid_size, 4)) * prob_action  # 4 possible actions

def next(state, action, grid_size):
    # Extract the x, y coordinates from the state
    x, y = state
    if (x == 0 and y == 0) or (x == grid_size - 1 and y == grid_size - 1):
        return state, 0  # If terminal, return the same state and zero reward

    # Compute the next state by adding the action
    next_state = (np.array(state) + action).tolist()
    next_x, next_y = next_state

    # If the next state is out of bounds, stay in the same state
    if next_x < 0 or next_x >= grid_size or next_y < 0 or next_y >= grid_size:
        next_state = state

    return next_state, -1


def graph(ax, image, title=None):
    ax.set_axis_off()  # Turn off the axis
    tb = Table(ax, bbox=[0, 0, 1, 1])  # Create a table object

    nrows, ncols = image.shape
    cell_size = 1.0 / max(nrows, ncols)  # Use one size for both width and height

    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, cell_size, cell_size, text=val, loc='center', facecolor='white')
    ax.add_table(tb)
    ax.set_title(title)


def policy_evaluation(state_values, policy, k, threshold=1e-4, gamma=gamma):
    """
    Evaluate the current policy to compute the value function.
    If k is None (or infinity), it runs until convergence based on the threshold.
    """
    iterations = 0
    actions = [np.array([0, -1]),  # Left
               np.array([-1, 0]),  # Up
               np.array([0, 1]),   # Right
               np.array([1, 0])]   # Down

    while True:
        new_state_values = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                value = 0
                for action, prob_action in zip(actions, policy[i, j]):
                    (next_i, next_j), reward = next([i, j], action, grid_size)
                    value += prob_action * (reward + gamma * state_values[next_i, next_j])
                new_state_values[i, j] = value
        
        # Check convergence condition based on the threshold
        if np.max(np.abs(new_state_values - state_values)) < threshold:
            print("Converged based on threshold")
            break

        # If k is provided and not None, stop after k iterations
        if k is not None and iterations >= k:
            print(f"Stopped after {k} iterations")
            break
        
        state_values = new_state_values.copy()
        iterations += 1
    return state_values, iterations


def figure_4_1():
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 6 unique plots in a 2x3 grid
    axs = axs.ravel()  # Flatten the 2D array of axes to 1D for easier iteration

    # Iteration counts for the different plots
    iteration_counts = [0, 1, 2, 3, 10, None]  # Last value is None to represent "infinity" (i.e., k=200)
    
    for i, iter_count in enumerate(iteration_counts):
        # Compute state values for the given number of iterations
        state_values = np.zeros((grid_size, grid_size))

        # If iter_count is None, run until convergence (i.e., k=200 case)
        if iter_count is None:
            state_values, _ = policy_evaluation(state_values, policy, k=None)  # No iteration limit, run until convergence
            sync_iteration = "\u221E"  # Use infinity symbol (∞)
        else:
            state_values, _ = policy_evaluation(state_values, policy, k=iter_count)
            sync_iteration = iter_count  # Normal iteration count
        
        values = state_values

        # Draw each plot on its corresponding subplot
        graph(axs[i], np.round(values, decimals=1), title=f'{sync_iteration} Iterations')

    # Save figure in the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    file_path = os.path.join(script_dir, 'figure_4_1_goat.png')  # Create file path in the script's directory
    print(file_path)
    plt.savefig(file_path)  # Save figure to the constructed path
    plt.tight_layout()  # Adjust layout to prevent overlap


if __name__ == '__main__':
    figure_4_1()
