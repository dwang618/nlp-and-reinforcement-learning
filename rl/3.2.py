import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Grid World Environment
gamma = 0.9
prob_action = 0.25

# next function for moving in the grid world
def next(state, action):
    special_states = {(0, 1): ([4, 1], 10), (0, 3): ([2, 3], 5)}
    state_tuple = tuple(state)

    # Check for special states A and B
    if state_tuple in special_states:
        return special_states[state_tuple]

    # Calculate the next state by updating x and y coordinates separately
    current_x, current_y = state[0], state[1]
    action_x, action_y = action[0], action[1]

    # Inefficient but explicit way of calculating the next state
    next_x = current_x + action_x
    next_y = current_y + action_y
    next_state = [next_x, next_y]

    # Check for boundary conditions
    if next_x < 0 or next_x >= 5 or next_y < 0 or next_y >= 5:
        return state, -1.0  # Stay in the same state with a reward of -1
    else:
        return next_state, 0  # Valid move with a reward of 0


# Create the MRP (Markov Reward Process) for the grid world
def create_grid_world_MRP():
    num_of_states = 25
    P = np.zeros((num_of_states, num_of_states))
    r = np.zeros((num_of_states, 1))

    for i in range(5):
        for j in range(5):
            index = i * 5 + j
            # Define the possible actions (left, up, right, down) without using ACTIONS array
            # Action: move left
            next_state, reward = next([i, j], [0, -1])
            next_i, next_j = next_state
            next_index = next_i * 5 + next_j
            P[index, next_index] += prob_action
            r[index] += prob_action * reward

            # Action: move up
            next_state, reward = next([i, j], [-1, 0])
            next_i, next_j = next_state
            next_index = next_i * 5 + next_j
            P[index, next_index] += prob_action
            r[index] += prob_action * reward

            # Action: move right
            next_state, reward = next([i, j], [0, 1])
            next_i, next_j = next_state
            next_index = next_i * 5 + next_j
            P[index, next_index] += prob_action
            r[index] += prob_action * reward

            # Action: move down
            next_state, reward = next([i, j], [1, 0])
            next_i, next_j = next_state
            next_index = next_i * 5 + next_j
            P[index, next_index] += prob_action
            r[index] += prob_action * reward

    return P, r

# Drawing utility to visualize the value function
def draw_image(image):
    fig, ax = plt.subplots()

    # Create a table to display values
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells to the table for each value in the image
    for i in range(nrows):
        for j in range(ncols):
            tb.add_cell(i, j, width, height, text=round(image[i, j], 2), loc='center', facecolor='white')

    # Add the table to the plot and display it
    ax.add_table(tb)
    ax.set_axis_off()
    plt.show()


# Function to calculate and visualize the value function
def grid_world_policy_evaluation():
    P, r = create_grid_world_MRP()
    v = policy_evaluation(P, r, gamma)
    
    # Reshape value function to fit the grid world for visualization
    grid_value = np.reshape(v, (5, 5))
    
    # Visualize the value function
    draw_image(np.round(grid_value, decimals=1))

# Policy evaluation function (unchanged from the original)
def policy_evaluation(P, r, gamma):
    bellman_operator = lambda v: r + gamma * P @ v
    v = np.random.randn(P.shape[0], 1)  # Initialize value function
    error = float('inf')
    while error > 1e-5:
        v_prev = v
        # Update value function for state s
        v = bellman_operator(v_prev)
        error = np.mean(np.abs(v - v_prev))
    return v

grid_world_policy_evaluation()
