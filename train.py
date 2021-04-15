import os
import logging
import sys

import numpy as np
import pandas as pd
import torch
from torchinfo import summary
import yaml

from src.agent import Agent
from src.tetris import Tetris


def get_logger(
        dirname: str = 'log',
        filename: str = 'log.log',
        name: str = __name__
    ) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    os.makedirs(dirname, exist_ok=True)
    logpath = os.path.join(dirname, filename)
    if os.path.exists(logpath):
        os.remove(logpath)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    handler = logging.FileHandler(logpath)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

def get_config(dirname : str = "configs", filename: str ="config.yaml") -> dict:
    with open(os.path.join(dirname, filename), "r+") as f:
        config = yaml.load(f)
    return config

experiment_name = sys.argv[1]

configdir = "configs"
configfile = experiment_name + '.yaml'
config = get_config(configdir, configfile)

logdir = config['logdir']
logfile = experiment_name + '.log'
weightdir = config["weightdir"]
weight_pretrained = ''
weightfile = experiment_name + '.pth'
resultdir = config['resultdir']
csvfile = experiment_name + '.csv'

num_episodes = config['num_episodes']
max_steps = config['max_steps']

height = config['height']
width = config['width']
block_size = config['block_size']

lr = config['learning_rate']
batch_size = config['batch_size']
gamma = config['gamma']
initial_eps = config['initial_eps']
final_eps = config['final_eps']
num_decay_epochs = config['num_decay_epochs']
capacity = config['capacity']
device = "cuda:0" if torch.cuda.is_available() else "cpu"

env = Tetris(num_rows=height, num_cols=width, block_size=block_size)
agent = Agent(
    lr=lr,
    batch_size=batch_size,
    gamma=gamma,
    initial_eps=initial_eps,
    final_eps=final_eps,
    num_decay_epochs=num_decay_epochs,
    capacity=capacity,
    device=device)
logger = get_logger(dirname=logdir, filename=logfile)

logger.info('-'*40 + ' Parameters ' + '-'*40)
logger.info(f'''
    # of episodes: {num_episodes}
    max steps: {max_steps}
    learning rate: {lr}
    batch size: {batch_size}
    gamma: {gamma}
    initial epsilon: {initial_eps}
    final epsilon: {final_eps}
    num_decay_epochs: {num_decay_epochs}
    memory capacity: {capacity}
    device: {device}
    model: '''
    )
logger.info(str(summary(agent.brain.model, (1, 1, 22, 10), verbose=0)) + '\n')

logger.info('-'*40 + ' Training log ' + '-'*40 + '\n')
history = {"steps": [], "total_reward": [], "cleared_lines": []}
max_step = 0
for episode in range(num_episodes):
    state = env.reset()[np.newaxis, ...].astype(np.float32)
    state = torch.from_numpy(state)

    total_reward = 0
    step = 0
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = np.stack(next_states)[:, np.newaxis].astype(np.float32)
        next_states = torch.from_numpy(next_states).to(device)

        next_state, action = agent.get_action(
            next_states, next_actions, episode)

        reward, done = env.step(action)
        total_reward += reward

        reward = torch.FloatTensor([reward]).to(device)
        action = torch.tensor([action], device=device)
        agent.memorize(state.to(device), action,
                       next_state.to(device), reward, done)
        agent.update_qval()

        state = next_state

        if done:
            message = f"Episode: {episode:4d}/{num_episodes:04d}\t" + \
                f"lr: {agent.brain.scheduler.get_last_lr()}\t" + \
                f"Step: {step:4d}\t" + \
                f"Cleared lines: {env.cleared_lines:3d}\t" + \
                f"Total Reward: {total_reward}"
            logger.info(message)
            history["steps"].append(step)
            history["total_reward"].append(total_reward)
            history["cleared_lines"].append(env.cleared_lines)

            if step >= max_step:
                os.makedirs(weightdir, exist_ok=True)
                max_step = step
                agent.save_model(os.path.join(weightdir, weightfile))
                logger.info("Model saved.")
            break
        step += 1
    agent.brain.scheduler.step()

history_df = pd.DataFrame(history)
os.makedirs(resultdir, exist_ok=True)
history_df.to_csv(os.path.join(resultdir, csvfile), index=False)

# plot_result(history, dirname=resultdir, prefix=experiment_name)
