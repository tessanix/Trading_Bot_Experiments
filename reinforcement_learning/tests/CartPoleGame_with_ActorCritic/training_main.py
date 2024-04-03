import gym
import numpy as np
from agent import Agent
# from utils import 

if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="human")

    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 10

    best_score = env.reward_range[0]
    score_history = []
    load_checkPoint = False

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(observation)
            # print(f"actions: {action}")
            observation_, reward, done, _, info = env.step(action)
            score += reward
            if not load_checkPoint:
                agent.learn(observation, reward, observation_, done)

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkPoint:
                agent.save_models()
        print(f'episode: {i}, score: {round(score,2)}, avg_score: {round(avg_score,2)}')


