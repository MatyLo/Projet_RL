import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pygame
from rl_environments.monty_hall_interactive import MontyHallInteractive
from rl_environments.monty_hall2_stepbystep import MontyHall2StepByStep
from rl_algorithms import QLearning, ValueIteration, SARSA
from visualization.monty_hall_visualizer import MontyHallVisualizer
import time

def run_monty_hall1(mode="human", agent_type="value_iteration"):
    episodes = 3  # Nombre d'épisodes fixé ici
    env = MontyHallInteractive()
    vis = MontyHallVisualizer()
    win_count = 0
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        if mode == "agent":
            if agent_type == "q_learning":
                agent = QLearning(env.state_space_size, env.action_space_size, learning_rate=0.1, gamma=0.9, epsilon=0.1)
                agent.train(env, num_episodes=1000)
            else:
                agent = ValueIteration(env.state_space_size, env.action_space_size, gamma=0.9)
                agent.train(env)
        while not done and vis.running:
            vis.render_monty_hall_state(env, episode_info={"Épisode": episode, "Victoires": win_count})
            if mode == "human":
                action = vis.get_human_action(env)
                if action is None:
                    vis.quit()
            else:
                action = agent.select_action(state)
                time.sleep(0.7)
            state, reward, done, info = env.step(action)
        if reward > 0:
            win_count += 1
        vis.render_monty_hall_state(env, episode_info={"Épisode": episode, "Victoires": win_count})
        pygame.time.wait(1200)
        if not vis.running:
            break
    vis.quit()

def run_monty_hall2(mode="human", agent_type="value_iteration"):
    episodes = 3  # Nombre d'épisodes fixé ici
    env = MontyHall2StepByStep()
    vis = MontyHallVisualizer()
    win_count = 0
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        if mode == "agent":
            if agent_type == "q_learning":
                agent = QLearning(env.state_space_size, env.action_space_size, learning_rate=0.1, gamma=0.9, epsilon=0.1)
                agent.train(env, num_episodes=2000)
            elif agent_type == "sarsa":
                agent = SARSA(env.state_space_size, env.action_space_size, learning_rate=0.1, gamma=0.9, epsilon=0.1)
                agent.train(env, num_episodes=2000)
            else:
                agent = ValueIteration(env.state_space_size, env.action_space_size, gamma=0.9)
                agent.train(env)
        while not done and vis.running:
            vis.render_monty_hall2_state(env, episode_info={"Épisode": episode, "Victoires": win_count})
            if mode == "human":
                action = vis.get_human_action_mh2(env)
                if action is None:
                    vis.quit()
            else:
                action = agent.select_action(state)
                time.sleep(0.7)
            state, reward, done, info = env.step(action)
        if reward > 0:
            win_count += 1
        vis.render_monty_hall2_state(env, episode_info={"Épisode": episode, "Victoires": win_count})
        pygame.time.wait(1200)
        if not vis.running:
            break
    vis.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['mh1', 'mh2'], default='mh1')
    parser.add_argument('--mode', choices=['human', 'agent'], default='human')
    parser.add_argument('--agent', choices=['value_iteration', 'q_learning', 'sarsa'], default='value_iteration')
    args = parser.parse_args()
    if args.env == 'mh1':
        run_monty_hall1(mode=args.mode, agent_type=args.agent)
    else:
        run_monty_hall2(mode=args.mode, agent_type=args.agent) 