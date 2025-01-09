import gymnasium as gym
import warnings

warnings.filterwarnings('ignore')

ENVIRONMENT = "Walker2d-v4"

env = gym.make(ENVIRONMENT, render_mode="human")

observation, info = env.reset()

env.render()

done, episode_length, total_reward = False, -0, 0

while not done:
    # 1. Select a random action
    # action = env.action_space.sample()
    action = (0, 0, 0, 0, 0, 0)

    # 2. Execute Selected Action
    observation, reward, terminated, truncated, info = env.step(action)

    # 3. keep track of the number of steps by incrementing variable
    episode_length += 1

    # 4. add the total reward
    total_reward += reward

    # If the episode has termindated or truncated exit loop
    done = terminated or truncated

print(f"Episode Length:\n\t{episode_length}\nTotal Reward:\n\t{total_reward}")

env.close()