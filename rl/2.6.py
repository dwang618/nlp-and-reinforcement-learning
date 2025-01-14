import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os

# Environment Class Definition (Handles rewards and transitions)
class BanditEnvironment:
    def __init__(self, k_arm=10):
        self.k = k_arm
        self.q_true = None
        self.best_action = np.argmax(self.q_true)

    def reset(self):
        # Real reward for each action
        self.q_true = np.random.randn(self.k)
        self.best_action = np.argmax(self.q_true)

    def step(self, action):
        # Generate reward for the chosen action
        reward = np.random.randn() + self.q_true[action]
        return reward

    def get_best_action(self):
        return self.best_action


# Epsilon-Greedy Agent
# Sample average bandit algorithm: Q(A)' = Q(A) + 1/N(A) * [R-Q(A)]
class EpsilonGreedyAgent:
    def __init__(self, epsilon, k_arm=10, step_size=0.1):
        self.k = k_arm
        self.epsilon = epsilon
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.step_size = step_size
        self.time = 0

    def reset(self):
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.time = 0

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.k))
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    def learn(self, action, reward):
        self.time += 1
        self.action_count[action] += 1
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]

# UCB Agent
# UCB Action Selector: argmax[Q_t(a) + c * sqrt( ln(t) / N_t(a))]
# Sample average bandit algorithm: Q(A)' = Q(A) + 1/N(A) * [R-Q(A)]
class UCBAgent:
    def __init__(self, UCB_param, k_arm=10, step_size=0.1, ):
        self.k = k_arm
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.step_size = step_size
        self.UCB_param = UCB_param
        self.time = 0

    def reset(self):
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.time = 0

    def act(self):
        # Increment the time step
        self.time += 1
        
        # Calculate the UCB estimation for each arm
        exploration_term = self.UCB_param * np.sqrt(np.log(self.time) / (self.action_count + 1e-5))
        UCB_estimation = self.q_estimation + exploration_term
        
        # Find the highest UCB estimation value
        q_best = np.max(UCB_estimation)
        
        # Find the indices of arms with the highest UCB estimation
        best_arm_indices = np.where(UCB_estimation == q_best)
        
        # Extract the array of indices from the tuple returned by np.where()
        best_arm_indices_array = best_arm_indices[0]
        
        # Randomly select one arm from the best arms
        selected_arm = np.random.choice(best_arm_indices_array)
        
        return selected_arm

    def learn(self, action, reward):
        self.action_count[action] += 1
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]


# Gradient Bandit Agent
class GradientBanditAgent:
    def __init__(self, k_arm=10, step_size=0.1):
        self.k = k_arm
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.step_size = step_size
        self.time = 0
        self.average_reward = 0

    def reset(self):
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.time = 0
        self.average_reward = 0

    def act(self):
        self.action_prob = np.exp(self.q_estimation) / np.sum(np.exp(self.q_estimation))
        return np.random.choice(np.arange(self.k), p=self.action_prob)

    def learn(self, action, reward):
        self.time += 1
        self.action_count[action] += 1

        # Update average reward
        # R_avg = R_avg + (R - R_avg) / t
        self.average_reward += (reward - self.average_reward) / self.time
        baseline = self.average_reward

        one_hot = np.zeros(self.k)
        one_hot[action] = 1

        # Update Q-values using policy gradient method
        # ΔQ(a) = α * (R - b) * (I{A=a} - π(a))
        # where α is step size, R is reward, b is baseline,
        # I{A=a} is indicator function (one-hot vector),
        # and π(a) is action probability
        self.q_estimation += self.step_size * (reward - baseline) * \
        (one_hot - self.action_prob)




# Optimistic Initialization Agent
class OptimisticInitializationAgent:
    def __init__(self, initial, k_arm=10, step_size=0.1):
        self.k = k_arm
        self.initial = initial
        self.q_estimation = np.zeros(self.k) + initial
        self.action_count = np.zeros(self.k)
        self.step_size = step_size

    # Add initial parameter to estimated reward
    def reset(self):
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)
        self.time = 0

    def act(self):
        # Find the highest Q-value estimation
        q_best = np.max(self.q_estimation)
        
        # Find the indices of arms with the highest Q-value estimation
        best_arm_indices = np.where(self.q_estimation == q_best)
        
        # Extract the array of indices from the tuple returned by np.where()
        best_arm_indices_array = best_arm_indices[0]
        
        # Randomly select one arm from the best arms
        selected_arm = np.random.choice(best_arm_indices_array)
        
        return selected_arm

    def learn(self, action, reward):
        self.action_count[action] += 1
        self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])    



def simulate(runs, time, agents, env):
    rewards = np.zeros((len(agents), runs, time))
    best_action_counts = np.zeros(rewards.shape)

    for i, agent in enumerate(agents):
        for r in trange(runs, desc=f'Agent {i + 1}/{len(agents)} Simulations'):
            env.reset()
            agent.reset()

            for t in range(time):
                action = agent.act()
                reward = env.step(action)
                agent.learn(action, reward)

                rewards[i, r, t] = reward
                if action == env.get_best_action():
                    best_action_counts[i, r, t] = 1

    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards

def figure_2_6(runs=2000, time=1000):
    env = BanditEnvironment()  # Same environment for all agents

    # Create agents with sample averages enabled
    labels = ['epsilon-greedy', 'gradient bandit', 'UCB', 'optimistic initialization']
    agents = []
    param_ranges = [
        np.arange(-7, -1, dtype=float),
        np.arange(-5, 2, dtype=float),
        np.arange(-4, 3, dtype=float),
        np.arange(-2, 3, dtype=float)
    ]

    # Epsilon-Greedy Agent
    for param in param_ranges[0]:
        actual_param = pow(2, param)
        agents.append(EpsilonGreedyAgent(epsilon=actual_param))

    # Gradient Bandit Agent
    for param in param_ranges[1]:
        actual_param = pow(2, param)
        agents.append(GradientBanditAgent(step_size=actual_param))

    # UCB Agent
    for param in param_ranges[2]:
        actual_param = pow(2, param)
        agents.append(UCBAgent(UCB_param=actual_param))

    # Optimistic Initialization Agent
    for param in param_ranges[3]:
        actual_param = pow(2, param)
        agents.append(OptimisticInitializationAgent(initial=actual_param, step_size=0.1))

    _, average_rewards = simulate(runs, time, agents, env)
    rewards = np.mean(average_rewards, axis=1)

    # Plotting
    colors = ['red', 'yellow', 'blue', 'black']
    start = 0

    for i in range(len(labels)):
        label = labels[i]
        param_range = param_ranges[i]
        color = colors[i]
        
        end = start + len(param_range)
        plt.plot(param_range, rewards[start:end], label=label, color=color)
        start = end

    # Create custom x-ticks and labels
    custom_ticks = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2])
    custom_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']

    plt.xticks(custom_ticks, custom_labels)
    plt.xlabel('Parameter Metric')
    plt.ylabel('Mean reward')
    plt.legend()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path for the PNG file
    png_path = os.path.join(script_dir, '2.6.png')

    # Save the figure
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    figure_2_6()
