from reinforcement_trainer import ReinforceTrainer, ReinforceAgent
import gymnasium as gym
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Hyperparameters for InvertedPendulum
n_episodes = 5000
hidden_size1 = 64
hidden_size2 = 64
learning_rate = 1e-4
gamma = 0.99

# Environment setup
env = gym.make('Walker2d-v4')
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Agent instantiation
agent = ReinforceAgent(obs_dim, action_dim, hidden_size1, hidden_size2, learning_rate, gamma)

# Train the agent
trainer = ReinforceTrainer(env, agent, n_episodes)
durations, lengths = trainer.train()

plt.plot(range(1, len(durations)+1), durations)

# Add labels and title
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Change by Episode')

plt.savefig("learning_curve.png")
