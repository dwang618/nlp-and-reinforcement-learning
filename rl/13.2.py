import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class ShortCorridor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 0  # Reset environment to the starting state

    def step(self, go_right):
        # Update state based on action (go_right: True/False)
        # Special transition dynamics: action may not always have expected effects
        if self.state in [0, 2]:  # Normal transition
            self.state += 1 if go_right else -1
        else:  # Reverse dynamics
            self.state += -1 if go_right else 1

        # Ensure state remains within bounds
        self.state = max(0, self.state)
        # If terminal state is reached, return reward 0, otherwise -1 for each step
        return (0, True) if self.state == 3 else (-1, False)


class ReinforceAgent:
    """
    Implements the REINFORCE algorithm without a baseline (Algorithm 13.1 in Sutton & Barto).
    """
    def __init__(self, alpha, gamma):
        self.theta = np.array([-1.47, 1.47])  # Policy parameter initialization
        self.alpha, self.gamma = alpha, gamma
        self.x = np.array([[0, 1], [1, 0]])  # Feature matrix (state-action encoding)
        self.rewards, self.actions = [], []  # Tracks rewards and actions for a trajectory

    def get_policy(self):
        """
        Computes the policy π(a|s; θ) using a softmax over state-action preferences.
        π(a|s; θ) = exp(h_a) / Σ_a' exp(h_a')
        """
        h = np.dot(self.theta, self.x)  # State-action preferences (linear in features)
        pmf = np.exp(h - np.max(h)) / np.sum(np.exp(h - np.max(h)))  # Softmax with stability adjustment
        epsilon = 0.05  # Ensures probabilities are bounded away from 0 and 1
        pmf = np.clip(pmf, epsilon, 1 - epsilon)
        return pmf

    def choose_action(self, reward):
        """
        Samples an action based on the policy π(a|s; θ) and stores the action.
        """
        if reward is not None:
            self.rewards.append(reward)  # Append reward from previous step
        pmf = self.get_policy()
        action = np.random.uniform() <= pmf[1]  # Action sampled: True (go_right) or False
        self.actions.append(action)
        return action

    def episode_end(self, last_reward):
        """
        Updates θ using the REINFORCE update rule:
        θ ← θ + α Σ_t γ^t G_t ∇_θ log π(a_t | s_t; θ)
        """
        self.rewards.append(last_reward)  # Add the terminal reward
        G = np.cumsum(self.rewards[::-1])[::-1]  # Compute returns G_t = Σ_k=t^T r_k
        for i, G_t in enumerate(G):
            grad_ln_pi = self.x[:, int(self.actions[i])] - np.dot(self.x, self.get_policy())
            self.theta += self.alpha * (self.gamma ** i) * G_t * grad_ln_pi  # Gradient ascent
        self.rewards, self.actions = [], []  # Reset for next episode


class ReinforceBaselineAgent(ReinforceAgent):
    """
    Implements the REINFORCE algorithm with a baseline b.
    """
    def __init__(self, alpha, gamma, alpha_w):
        super().__init__(alpha, gamma)
        self.alpha_w, self.w = alpha_w, 0  # Baseline parameters: step size and baseline value

    def episode_end(self, last_reward):
        """
        Updates θ and w using the REINFORCE update rule with baseline:
        θ ← θ + α Σ_t γ^t (G_t - b(s_t)) ∇_θ log π(a_t | s_t; θ)
        w ← w + α_w Σ_t γ^t (G_t - w)
        """
        self.rewards.append(last_reward)
        G = np.cumsum(self.rewards[::-1])[::-1]  # Compute returns G_t
        for i, G_t in enumerate(G):
            delta = G_t - self.w  # Temporal difference using baseline
            self.w += self.alpha_w * (self.gamma ** i) * delta  # Update baseline parameter w
            grad_ln_pi = self.x[:, int(self.actions[i])] - np.dot(self.x, self.get_policy())
            self.theta += self.alpha * (self.gamma ** i) * delta * grad_ln_pi  # Gradient ascent
        self.rewards, self.actions = [], []  # Reset for next episode


def trial(num_episodes, agent_generator):
    """
    Runs a single trial of a given agent in the environment over multiple episodes.
    """
    env = ShortCorridor()
    agent = agent_generator()
    rewards = np.zeros(num_episodes)

    for ep in range(num_episodes):
        total_reward, reward = 0, None
        env.reset()

        while True:
            action = agent.choose_action(reward)  # Select action
            reward, done = env.step(action)  # Take step in environment
            total_reward += reward
            if done:
                agent.episode_end(reward)  # Update agent parameters at end of episode
                break

        rewards[ep] = total_reward
    return rewards


def figure_13_2():
    """
    Reproduces Figure 13.2 from Sutton & Barto:
    Compares REINFORCE with and without baseline in the short corridor example.
    """
    num_trials, num_episodes, alpha, gamma = 100, 1000, 2e-4, 1
    agents = [
        lambda: ReinforceAgent(alpha=alpha, gamma=gamma),  # Without baseline
        lambda: ReinforceBaselineAgent(alpha=alpha * 10, gamma=gamma, alpha_w=alpha * 100),  # With baseline
    ]
    labels = ['Reinforce without baseline', 'Reinforce with baseline']
    rewards = np.array([[trial(num_episodes, agent) for _ in tqdm(range(num_trials))] for agent in agents])

    # Directly plotting the results
    baseline = -11.6
    plt.plot(np.arange(1, num_episodes + 1), baseline * np.ones(num_episodes), ls='dashed', color='red') 
    for reward, label in zip(rewards, labels):
        plt.plot(np.arange(1, num_episodes + 1), reward.mean(axis=0), label=label)
    plt.ylabel('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.legend(loc='lower right')

    # Ensure desired ticks on y-axis
    plt.yticks(np.arange(-90, -5, 10))  # Set y-axis ticks from -100 to 0 with a step of 10

    plt.show()

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '13.2ex.png')
    plt.savefig(file_path)


if __name__ == '__main__':
    figure_13_2()
