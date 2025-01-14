import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# Set up environment and policy classes
class Environment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.trajectory = []
        self.done = False
        return 0  # initial state

    def step(self):
        action = np.random.binomial(1, 0.5)
        self.trajectory.append(action)
        if action == 1:
            self.done = True
            return 0  # End state reward
        if np.random.binomial(1, 0.9) == 0:
            self.done = True
            return 1  # Reward on termination
        return None  # Continue without reward

class Agent:
    def __init__(self, env):
        self.env = env
        self.rewards_ois = []
        self.rewards_wis = []

    def learn(self):
        # Calculate importance sampling ratios
        if self.env.trajectory[-1] == 1:
            sampling_ratio = 0                  # Addresing trajectories not consistent with policy
        else:
            sampling_ratio = 1.0 / pow(0.5, len(self.env.trajectory))
        return sampling_ratio

    def run(self, episodes):
        """
        Simulate a specified number of episodes, tracking rewards under both Ordinary
        and Weighted Importance Sampling methods. Each episode calculates cumulative
        estimates for each method.
        """
        rewards_ois = []
        rewards_wis = []
        cumulative_sampling_ratio = 0
        cumulative_weighted_reward = 0

        for episode in range(episodes):
            self.env.reset()
            total_reward = 0

            # Run the episode until it reaches terminal state
            while not self.env.done:
                reward = self.env.step()
                if reward is not None:
                    total_reward = reward
                    sampling_ratio = self.learn()
            cumulative_sampling_ratio += sampling_ratio
            cumulative_weighted_reward += sampling_ratio * reward

            # Ordinary Importance Sampling (OIS): rho * reward
            rewards_ois.append(sampling_ratio * reward)

            # Weighted Importance Sampling
            if cumulative_sampling_ratio != 0:
                weighted_reward = cumulative_weighted_reward / cumulative_sampling_ratio
            else:
                weighted_reward = 0
            rewards_wis.append(weighted_reward)

        # Calculate cumulative OIS averages over episodes
        self.rewards_ois = np.cumsum(rewards_ois) / np.arange(1, episodes + 1)
        self.rewards_wis = np.array(rewards_wis)


def figure_5_4():
    env = Environment()
    ois_results = []
    wis_results = []

    for run in range(10):
        agent = Agent(env)
        agent.run(int(1e6))
        ois_results.append(agent.rewards_ois)
        wis_results.append(agent.rewards_wis)

    # Plot results
    for i in range(10):
        plt.plot(ois_results[i], color='blue', alpha=0.5, label='OIS' if i == 0 else "")
        plt.plot(wis_results[i], color='orange', alpha=0.5, label='WIS' if i == 0 else "")

    plt.xlabel('Episode Index')
    plt.xscale('log')
    plt.legend(loc='best')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '5_4figure_.png')
    plt.savefig(file_path)
    plt.show()
    plt.close()

if __name__ == '__main__':
    figure_5_4()
