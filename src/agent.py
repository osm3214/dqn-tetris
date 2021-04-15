import random
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR

from .models import TetrisNet


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.memory: List[Transition] = []
        self.position: int = 0

    def push(self, *args: Tuple) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class Brain(object):
    def __init__(
            self,
            lr: float = 1e-4,
            batch_size: int = 512,
            gamma: float = 0.99,
            initial_eps: float = 1.0,
            final_eps: float = 1e-3,
            num_decay_epochs: int = 2000,
            capacity: int = 30000,
            device: str = "cpu"
    ) -> None:
        self.lr = lr
        self.batch_size = batch_size

        self.gamma = gamma
        self.capacity = capacity
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.num_decay_epochs = num_decay_epochs

        self.memory = ReplayMemory(capacity)
        self.model = TetrisNet()
        self.model.to(device)
        # self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[2500, 3500, 4500], gamma=0.1)
        self.criterion = nn.MSELoss()

    def replay(self) -> None:
        if len(self.memory) < self.capacity / 10:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state)
        reward_batch = torch.stack(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = batch.done

        self.model.train()
        q_values = self.model(state_batch)

        self.model.eval()
        with torch.no_grad():
            next_prediction_batch = self.model(next_state_batch)

        exp_qvals = torch.cat(
            tuple(reward if done else reward + self.gamma * prediction
                  for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, exp_qvals)

        loss.backward()
        self.optimizer.step()

    def get_action(
            self,
            next_states: torch.Tensor,
            next_actions: tuple,
            episode: int
        ) -> tuple:

        eps = self.final_eps + \
            (max(self.num_decay_epochs - episode, 0) *
             (self.initial_eps - self.final_eps) / self.num_decay_epochs)
        if eps < np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(next_states)[:, 0]
                index = torch.argmax(predictions).item()
        else:
            index = random.randint(0, len(next_states) - 1)
        next_state = next_states[index, :]
        next_action = next_actions[index]

        return next_state, next_action

    def get_action_eval(
            self,
            next_states: torch.Tensor,
            next_actions: tuple) -> tuple:
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        return action

    def load_model(self, path: str, device=None) -> None:
        if device is not None:
            self.model.load_state_dict(torch.load(path, map_location=device))
        else:
            self.model.load_state_dict(torch.load(path))

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)


class Agent(object):
    def __init__(
        self,
        lr: float = 1e-4,
        batch_size: int = 512,
        gamma: float = 0.99,
        initial_eps: float = 1.0,
        final_eps: float = 1e-3,
        num_decay_epochs: int = 2000,
        capacity: int = 30000,
        device: str = "cpu"
    ) -> None:
        self.brain: Brain = Brain(
            lr=lr,
            batch_size=batch_size,
            gamma=gamma,
            initial_eps=initial_eps,
            final_eps=final_eps,
            num_decay_epochs=num_decay_epochs,
            capacity=capacity,
            device=device
        )

    def update_qval(self) -> None:
        self.brain.replay()

    def get_action(
            self,
            next_states: torch.Tensor,
            next_actions: tuple,
            episode: int) -> tuple:
        next_state, next_action = self.brain.get_action(next_states, next_actions, episode)
        return next_state, next_action

    def get_action_eval(
            self,
            next_states:torch.Tensor,
            next_actions: tuple) -> tuple:
        action = self.brain.get_action_eval(next_states, next_actions)
        return action

    def memorize(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            state_next: torch.Tensor,
            reward: torch.Tensor,
            done: list) -> None:
        self.brain.memory.push(state, action, state_next, reward, done)

    def load_model(self, path: str, device=None) -> None:
        self.brain.load_model(path, device)

    def save_model(self, path: str) -> None:
        self.brain.save_model(path)
