import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Environment Class Definition (Handles Rewards and Transitions)
class BanditEnvironment:
    # Ten armed bandit configuration
    def __init__(self, num_arms=10):
        self.num_arms = num_arms
        self.q_true = None  

    # Randomize true reward for each arm
    def reset(self):
        self.q_true = np.random.randn(self.num_arms)

    # Set true reward for each arm
    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        return reward

# UCB Agent Class Definition
class UCBAgent:
    def __init__(self, UCB_param, num_arms=10, step_size=0.1):
        self.num_arms = num_arms
        self.step_size = step_size
        self.UCB_param = UCB_param
        self.q_estimate = np.zeros(self.num_arms)
        self.actions_counts = np.zeros(self.num_arms)
        self.time = 0

    def reset(self):
        self.q_estimate = np.zeros(self.num_arms)
        self.actions_counts = np.zeros(self.num_arms)
        self.time = 0

    def act(self):
        self.time += 1
        if np.min(self.actions_counts) == 0:
            return np.argmin(self.actions_counts)  # Ensure all arms are tried at least once
        
        # UCB Action Selector: argmax[Q_t(a) + c * sqrt( ln(t) / N_t(a))]
        UCB_estimation = self.q_estimate + self.UCB_param * np.sqrt(np.log(self.time) / (self.actions_counts + 1e-5))
        return np.argmax(UCB_estimation)
    
    # Sample average bandit algorithm: Q(A)' = Q(A) + 1/N(A) * [R-Q(A)]
    def learn(self, action, reward):
        self.actions_counts[action] += 1
        self.q_estimate[action] += (reward - self.q_estimate[action]) / self.actions_counts[action]
        
# Epsilon Greedy Agent with Sample Averages
class EpsilonGreedyAgent:
    def __init__(self, epsilon, num_arms=10, step_size=0.1):
        self.epsilon = epsilon
        self.num_arms = num_arms
        self.step_size = step_size
        self.q_estimate = np.zeros(self.num_arms)
        self.actions_counts = np.zeros(self.num_arms)

    def reset(self):
        self.q_estimate = np.zeros(self.num_arms)
        self.actions_counts = np.zeros(self.num_arms)

    def act(self):
        # Greedy choice algorithm
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_arms)
        return np.argmax(self.q_estimate)

    # Sample average bandit algorithm: Q(A)' = Q(A) + 1/N(A) * [R-Q(A)]
    def learn(self, action, reward):
        self.actions_counts[action] += 1
        self.q_estimate[action] += (reward - self.q_estimate[action]) / self.actions_counts[action]

# Figure 2.4 Plot
def figure_2_4():
    env = BanditEnvironment()  # Same environment for both agents

    # Create agents with sample averages enabled
    agents = []
    agents.append(UCBAgent(UCB_param=2))  # UCB agent
    agents.append(EpsilonGreedyAgent(epsilon=0.1))  # Epsilon-greedy agent

    # Run the simulation
    rewards = np.zeros((len(agents), 2000, 1000))
    
    for i, agent in enumerate(agents):
        for run in tqdm(range(2000), desc=f"Running Agent {i+1}/{len(agents)}"):
            env.reset()
            agent.reset()

            for t in range(1000):
                action = agent.act()
                reward = env.step(action)
                agent.learn(action, reward)

                rewards[i, run, t] = reward

    average_rewards = rewards.mean(axis=1)

    # Plot the results
    plt.plot(average_rewards[0], label='Upper Confidence Bound')
    plt.plot(average_rewards[1], label='0.1 Epsilon Greedy Action')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path for the PNG file
    png_path = os.path.join(script_dir, '2.4.png')

    # Save the figure
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.show()

# Run the figure generation
figure_2_4()
