from Agent2 import Agent2
from ShowerEnv import ShowerEnv
import matplotlib.pyplot as plt
import numpy as np

env = ShowerEnv(100, 5, 10, 35)

agent = Agent2(gamma=0.995, lr=0.01, n_actions=20, eps_dec=0.9, eps_start=1.0, eps_end=0.01)


episodes = 10_000
scores = []

if __name__ == "__main__":
    print("Teaching Agent")

    for step in range(episodes):
        # Initialize environment
        done = False
        state = env.reset()
        score = 0

        # training the agent
        while not done:
            # Have agent select an action
            action = agent.choose_actions(state)

            # Based on the selected action make a step in the env
            next_state, reward, done = env.step(action)

            # Based on the results from the action, update agent
            agent.learn(state, action, reward, next_state)

            # update episode's score
            score += reward

            # update the environments state
            state = next_state
        
        # Print after ever 100 steps
        if step % 100 == 0: print(f"Step: {step} | Score: {score}")
        # at the end of each episode, add the results to the list of scores
        scores.append(score)


    plt.plot(scores)
    plt.xlabel("Iterations")
    plt.ylabel("Reward")

    plt.savefig("ShowerCustomEnv/plot.png")