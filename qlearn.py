import numpy as np
import tensorflow as tf

from copy import deepcopy

import cv2
import gym
import random
import time

import history
import network
import state

import config as _config

class qlearn(object):
    def __init__(self, config):
        self.total_steps = 0

        self.env = gym.make(config.get('game'))

        self.input_shape = config.get('input_shape')
        self.state_steps = config.get('state_steps')

        self.batch_size = config.get('batch_size')

        self.current_state = state.state(self.input_shape, self.state_steps)

        self.actions = self.env.action_space.n
        config.put('actions', self.actions) # used in network to create N outputs, one per action

        self.epsilon_start = config.get('epsilon_start')
        self.epsilon_end = config.get('epsilon_end')
        self.initial_explore_steps = config.get('initial_explore_steps')
        self.total_explore_steps = config.get('total_explore_steps')
        self.epsilon = self.epsilon_start
        self.alpha = config.get('q_alpha')
        self.discount_gamma = config.get('discount_gamma')

        self.history = history.history(config.get('history_size'))

        output_path = config.get('output_path')
        output_path += '/run.%d' % (time.time())
        self.summary_writer = tf.summary.FileWriter(output_path)
        config.put('summary_writer', self.summary_writer) # used in network

        self.main = network.network(config)

    def new_state(self, state):
        state = 0.2126 * state[:, :, 0] + 0.7152 * state[:, :, 1] + 0.0722 * state[:, :, 2]

        state = state.astype(np.float32)
        res = cv2.resize(state, (self.input_shape[0], self.input_shape[1]))
        #res /= 255.

        res = np.reshape(res, self.input_shape)

        self.current_state.push_tensor(res)
        return deepcopy(self.current_state)

    def reset(self):
        self.current_state = state.state(self.input_shape, self.state_steps)
        obs = self.env.reset()
        return self.new_state(obs)

    def get_action(self, s):
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.actions)
        else:
            action_idx = self.get_predicted_action(s)

        return action_idx

    def get_predicted_action(self, s):
        q = self.main.predict([s.read()])
        return np.argmax(q)

    def store(self, data):
        if self.epsilon > self.epsilon_end and self.total_steps > self.initial_explore_steps:
            self.epsilon -= (self.epsilon_start - self.epsilon_end) / self.total_explore_steps

        self.history.append(data)

    def train(self):
        batch = self.history.sample(self.batch_size)

        states_shape = (len(batch), self.state_steps, self.input_shape[0], self.input_shape[1])
        states = np.ndarray(shape=states_shape)
        next_states = np.ndarray(shape=states_shape)

        q_shape = (len(batch), self.actions)
        qvals = np.ndarray(shape=q_shape)
        next_qvals = np.ndarray(shape=q_shape)

        for idx, e in enumerate(batch):
            s, a, r, sn, done = e

            states[idx] = s.read()
            next_states[idx] = sn.read()

        qvals = self.main.predict(states)
        next_qvals = self.main.predict(next_states)

        for idx, e in enumerate(batch):
            s, a, r, sn, done = e

            qmax_next = np.amax(next_qvals[idx])
            if done:
                qmax_next = 0

            current_qa = qvals[idx][a]
            qsa = (1. - self.alpha) * current_qa + self.alpha * (r + self.discount_gamma * qmax_next)
            qvals[idx][a] = qsa

        self.main.train(states, qvals)

    def episode(self):
        s = self.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.get_action(s)

            obs, reward, done, info = self.env.step(action)
            sn = self.new_state(obs)
            self.store((s, action, reward, sn, done))
            self.total_steps += 1

            if self.history.size() > self.batch_size:
                self.train()

            s = sn
            total_reward += reward

        return total_reward

    def run(self, num_episodes):
        rewards = history.history(100)
        for i in xrange(num_episodes):
            reward = self.episode()

            rewards.append(reward)

            print("{:4d}: steps: {:6d}, epsilon: {:.2f}, reward: {:2f}, average reward per {:3d} last episodes: {:.2f}".format(
                i, self.total_steps, self.epsilon,
                reward, rewards.size(), np.mean(rewards.whole())))
