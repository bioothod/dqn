import numpy as np
import tensorflow as tf

from copy import deepcopy

import collections
import cv2
import gym
import random
import time

import history
import network
import state

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class qlearn(object):
    def __init__(self, config):
        self.total_steps = 0

        self.env = gym.make(config.get('game'))
        self.env = MaxAndSkipEnv(self.env)
        self.env = FireResetEnv(self.env)

        self.input_shape = config.get('input_shape')
        self.state_steps = config.get('state_steps')

        self.batch_size = config.get('batch_size')

        self.train_interval = config.get('train_interval')
        self.start_train_after_steps = config.get('start_train_after_steps')

        self.update_follower_steps = config.get('update_follower_steps')

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

        with tf.variable_scope('main') as vscope:
            self.main = network.network('main', config)
        with tf.variable_scope('follower') as vscope:
            self.follower = network.network('follower', config)

        self.follower.import_params(self.main.export_params(), 0)

    def new_state(self, state):
        state = 0.2126 * state[:, :, 0] + 0.7152 * state[:, :, 1] + 0.0722 * state[:, :, 2]

        state = state.astype(np.float32)
        res = cv2.resize(state, (self.input_shape[0], self.input_shape[1]))
        res /= 255.

        res = np.reshape(res, self.input_shape)

        self.current_state.push_tensor(res)
        return deepcopy(self.current_state)

    def reset(self):
        self.current_state = state.state(self.input_shape, self.state_steps)
        obs = self.env.reset()
        return self.new_state(obs)

    def get_action(self, s):
        if np.random.rand() <= self.epsilon:
            action_idx = self.env.action_space.sample()
        else:
            action_idx = self.get_predicted_action(s)

        return action_idx

    def get_predicted_action(self, s):
        return np.argmax(self.follower.predict([s.read()]), axis=1)

    def store(self, data):
        if self.epsilon > self.epsilon_end and self.total_steps > self.initial_explore_steps:
            self.epsilon -= (self.epsilon_start - self.epsilon_end) / self.total_explore_steps

        self.history.append(data)

    def train(self):
        batch = self.history.sample(self.batch_size * self.train_interval)

        states_shape = (len(batch), self.input_shape[0], self.input_shape[1], self.input_shape[2]*self.state_steps)
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
        next_qvals = self.follower.predict(next_states)

        #print("q1: {}".format(qvals))

        for idx, e in enumerate(batch):
            s, a, r, sn, done = e

            qmax_next = np.amax(next_qvals[idx])
            if done:
                qmax_next = 0

            current_qa = qvals[idx][a]
            # clip Q value to [-1;+1] interval
            #qsa = min(1, max(-1, (1. - self.alpha) * current_qa + self.alpha * (r + self.discount_gamma * qmax_next)))
            qsa = (1. - self.alpha) * current_qa + self.alpha * (r + self.discount_gamma * qmax_next)
            qvals[idx][a] = qsa

        #print("q2: {}".format(qvals))
        self.main.train(states, qvals)
        if self.main.train_steps % self.update_follower_steps == 0:
            self.follower.import_params(self.main.export_params(), 0)

    def episode(self):
        s = self.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = self.get_action(s)

            obs, reward, done, info = self.env.step(action)
            sn = self.new_state(obs)
            self.store((s, action, reward, sn, done))
            self.total_steps += 1
            steps += 1

            if steps % self.train_interval == 0 and self.total_steps > self.start_train_after_steps:
                self.train()

            s = sn
            total_reward += reward

        self.main.update_rewards([total_reward])
        return total_reward

    def run(self, num_episodes):
        rewards = history.history(100)
        for i in xrange(num_episodes):
            reward = self.episode()

            rewards.append(reward)

            print("{:4d}: steps: {:6d}, epsilon: {:.2f}, reward: {:2.1f}, average reward per {:3d} last episodes: {:.2f}".format(
                i, self.total_steps, self.epsilon,
                reward, rewards.size(), np.mean(rewards.whole())))
