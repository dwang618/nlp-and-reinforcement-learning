import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Initialize state values
state_nodes = ['A', 'B', 'C', 'D', 'E']
values_array = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
true_values = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])

# Parameters For Individual Use Cases
temporal_diff_alpha = [0.15, 0.1, 0.05]
monte_carlo_alpha = [0.01, 0.02, 0.03, 0.04]

# Temporal Difference Learning Implementation
def TD(values, alpha=0.1):
    state = 3
    while state not in [0, 6]:
        next_state = state - 1 if np.random.rand() < 0.5 else state + 1
        # V(s) <- V(s) + alpha(V(s') - V(s))
        values[state] += alpha * (values[next_state] - values[state])
        state = next_state

# Monte Carlo Implementation
def MC(values, alpha=0.1):
    state = 3
    trajectory = []
    while state not in [0, 6]:
        trajectory.append(state)
        state = state - 1 if np.random.rand() < 0.5 else state + 1
    returns = 1 if state == 6 else 0
    for s in trajectory:
        # V(s) <- V(s) + alpha(return - V(s))
        values[s] += alpha * (returns - values[s])

def plot_state_values():
    episodes = [0, 1, 10, 100]
    values = values_array.copy()
    
    for ep in range(101):
        if ep in episodes:
            plt.plot(state_nodes, values[1:6], label=None)  # No label for the line
            # Annotate the episode number at the last point of the curve
            plt.text(4.5, values[5], f'{ep} episodes', fontsize=10, ha='left')

        TD(values)

    # Plot the true values curvex``
    plt.plot(true_values[1:6], color='black', linestyle='--')
    # Annotate the true values line
    plt.text(4.5, true_values[5], 'True values', fontsize=10, ha='left', color='black')

    plt.xlabel('State')
    plt.ylabel('Estimated Value')


def plot_RMS():
    for alpha in temporal_diff_alpha + monte_carlo_alpha:
        errors = np.zeros(101)
        method = 'TD' if alpha in temporal_diff_alpha else 'MC'
        linestyle = 'solid' if method == 'TD' else 'dashdot'
        for _ in tqdm(range(100), desc=f'{method} α={alpha:.02f}'):
            values = values_array.copy()
            for ep in range(101):

                # RMSE Formula Between True Value and Estimated Value
                errors[ep] += np.sqrt(np.mean((true_values - values) ** 2))
                (TD if method == 'TD' else MC)(values, alpha)

        plt.plot(errors / 100, linestyle=linestyle, label=f'{method}, α={alpha:.02f}')

    # Add a legend for the line styles
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], linestyle='solid', color='black', label='TD (Temporal Difference)'),
        Line2D([0], [0], linestyle='dashdot', color='black', label='MC (Monte Carlo)')
    ]
    plt.legend(handles=custom_lines, loc='upper right')
    plt.xlabel('Episodes')
    plt.ylabel('Root Mean Square Error')

def main():
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plot_state_values()

    plt.subplot(2, 1, 2)
    plot_RMS()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '6_2ex.png')
    plt.savefig(file_path)
    
    plt.show()

if __name__ == '__main__':
    main()
