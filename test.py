import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from src.tetris import Tetris
from src.agent import Agent


experiment_name = sys.argv[1]

device = "cpu"
weightdir = "models"
weight_pretrained = experiment_name + ".pth"
resultdir = "results"
giffile = experiment_name + '.gif'

env = Tetris()
env.reset()
agent = Agent(device=device)
agent.load_model(os.path.join(weightdir, weight_pretrained), device=device)

fig = plt.figure()
plt.axis('off')
img = env.render()
# img = plt.imshow(img_array)
# fig.suptitle('step: 0')

imgs = []
step = 0
imgs.append([plt.imshow(img)])
while True:
    next_steps = env.get_next_states()
    next_actions, next_states = zip(*next_steps.items())
    next_states = np.stack(next_states)[:, np.newaxis].astype(np.float32)
    next_states = torch.from_numpy(next_states).to(device)
    action = agent.get_action_eval(next_states.to(device), next_actions)
    reward, done = env.step(action)

    traced_cell = torch.jit.trace(agent.brain.model, next_states)

    img = env.render()
    # fig.suptitle('step: {}'.format(step + 1))
    step = step + 1
    imgs.append([plt.imshow(img)])
    # plt.pause(0.50)

    if done:
        break

ani = animation.ArtistAnimation(fig, imgs, interval=500)
ani.save(os.path.join(resultdir, giffile))
traced_cell.save('models/tetrisnet.pt')
