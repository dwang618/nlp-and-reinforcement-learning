import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
import os

grid_size = 5
prob_action = 0.25
gamma = 0.9

A_Reward = 10
B_Reward = 5

def next(state, action):
    # Extract the x, y coordinates from the state
    x, y = state

    # Special reward conditions for designated states
    if state == [0, 1]:
        return [4, 1], A_Reward  # Move to [4, 1] with reward A_Reward (10)
    if state == [0, 3]:
        return [2, 3], B_Reward   # Move to [2, 3] with reward B_Reward (5)

    # Compute the next state by adding the action
    next_state = (np.array(state) + action).tolist()
    next_x, next_y = next_state

    # Check if next state is out of bounds
    if next_x < 0 or next_x >= grid_size or next_y < 0 or next_y >= grid_size:
        reward = -1.0  # Penalty for hitting the boundary
        next_state = state  # Remain in the same state
    else:
        reward = 0  # No reward for regular moves within bounds

    return next_state, reward



def draw_image(ax, image, title=None):
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    cell_size = 1.0 / max(nrows, ncols)

    # Add cells with values from the image array
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, cell_size, cell_size, text=str(val), loc='center', facecolor='white')
    
    ax.add_table(tb)
    ax.set_title(title)



def value_iteration(value, gamma, threshold=1e-4):
    """
    Value iteration algorithm to compute the value function until convergence.
    """
    iterations = 0
    actions = [np.array([0, -1]),  # Left
               np.array([-1, 0]),  # Up
               np.array([0, 1]),   # Right
               np.array([1, 0])]   # Down

    while True:
        new_value = np.zeros_like(value)
        for i in range(grid_size):
            for j in range(grid_size):
                values = []
                for action in actions:
                    (next_i, next_j), reward = next([i, j], action)
                    #Bellman Eq for v*
                    values.append(reward + gamma * value[next_i, next_j])

                #Optimal action under greedy policy
                new_value[i, j] = np.max(values)
        
        #Convergence check
        if np.max(np.abs(new_value - value)) < threshold:
            break

        value = new_value.copy()
        iterations += 1
    return new_value, iterations


def figure_3_5():
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))  # Single plot for the value iteration result

    # Initialize value grid and run value iteration until convergence
    initial_value = np.zeros((grid_size, grid_size))
    values, _ = value_iteration(initial_value, gamma)

    # Display the result on the plot
    draw_image(axs, np.round(values, decimals=1), title="Value Iteration Result: v*")
    
    # Save figure in the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'figure_3_5.png')
    plt.savefig(file_path)
    print(file_path)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    figure_3_5()
