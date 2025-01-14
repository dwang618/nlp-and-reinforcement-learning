import matplotlib.pyplot as plt
import numpy as np
import os

# Ten Armed Bandit Class Definition
class Ten_Armed_Bandit:
    num_arms = 10
    step_size = 0.1
    
    # Initialize bandit with estimated value and greedy probability coefficient 
    def __init__(self, epsilon, initial):
        self.epsilon = epsilon
        self.initial = initial

    # Selection strategy for greedy choice or max estimated value
    def act(self, q_estimate):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_arms)
        return np.argmax(q_estimate)

    # Generate a random reward to add to the true value of an arm
    def step(self, action, q_true):
        reward = np.random.randn() + q_true[action]
        return reward

    # Update counter arrays and estimated value based off of action
    def learn(self, action, reward, q_estimate, action_rewards, action_counts):
        action_rewards[action] += reward
        action_counts[action] += 1

        # Rule (2.3): NewEstimate <- OldEstimate  + StepSize[Target - OldEstimate]
        q_estimate[action] += self.step_size * (reward - q_estimate[action])

# Define parameters
runs, time = 2000, 1000
bandits = [Ten_Armed_Bandit(epsilon=0, initial=5), Ten_Armed_Bandit(epsilon=0.1, initial=0)]

# Initialize arrays to store best actions and action_rewards
action_counts = np.zeros((len(bandits), runs, time)) 
action_rewards = np.zeros((len(bandits), runs, time))

# Simulation
for i, bandit in enumerate(bandits):
    for run in range(runs):
        # Manually reset variables at the start of each run
        q_true = np.random.randn(bandit.num_arms)  # True action values
        q_estimate = np.zeros(bandit.num_arms) + bandit.initial  # Estimated action values
        local_action_rewards = np.zeros(bandit.num_arms)  # Local rewards
        local_action_counts = np.zeros(bandit.num_arms)  # Local counts
        optimal_action = np.argmax(q_true)  # Best action based on true values

        for t in range(time):
            # Select an action based on current estimates
            action = bandit.act(q_estimate)
            
            # Get reward for the selected action
            reward = bandit.step(action, q_true)
            
            # Update estimations using learn method
            bandit.learn(action, reward, q_estimate, local_action_rewards, local_action_counts)
            
            # Store the reward
            action_rewards[i, run, t] = reward
            
            # Check if the action is optimal
            if action == optimal_action:
                action_counts[i, run, t] = 1

# Aggregate results over runs
action_counts_mean = action_counts.sum(axis=1) / runs

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(action_counts_mean[0], label=r'$\epsilon = 0, q = 5$')
plt.plot(action_counts_mean[1], label=r'$\epsilon = 0.1, q = 0$')
plt.xlabel('Steps')
plt.ylabel('% optimal action')
plt.legend(loc='lower right')

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the full path for the PNG file
png_path = os.path.join(script_dir, '2.3.png')

# Save the figure
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.close()

