from Agent import Agent
from GridworldEnv import GridEnv
import matplotlib.pyplot as plt

# Define Hyperparameters
gamma, lr, n_actions = 0.995, 0.01, 4
eps_dec, eps_start, eps_end = 0.9, 1.0, 0.1

agent = Agent(lr, gamma, n_actions, eps_start, eps_end, eps_dec)
env = GridEnv(10)

if __name__ == "__main__":
    scores = []
    n_eps = 10000  


    for i in range(n_eps):
        done = False
        state = env.reset()
        score = 0
    
        while not done:
            action = agent.choose_actions(state)
            state_, reward, done = env.step(action)
            agent.learn(state, action, reward, state_)
            score += reward
            state = state_
        
        scores.append(score)

        if i % 500 == 0: print(f"Episode: {i} | Score: {score}")


    plt.plot(scores)     
    plt.xlabel('Episodes')
    plt.ylabel('Reward')

    plt.savefig("training_chart.jpg")