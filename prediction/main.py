import gymnasium as gym
from agent import Agent


if __name__ == '__main__':
    env = gym.make('Blackjack-v1', sab=True)
    agent = Agent()
    n_episodes = 500000
    for i in range(n_episodes):
        if i % 50000 == 0:
            print('starting episode ', i)
        observation, info = env.reset()
        done = False
        while not done:
            action = agent.policy(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            agent.memory.append((observation, reward))
            observation = observation_
            if terminated or truncated:
                done = True
        agent.update_V()
    print(agent.V[(21, 3, True)])
    print(agent.V[(4, 1, False)])