import numpy as np
from copy import deepcopy

import cv2
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import config
import qlearn

if __name__ == '__main__':
    c = config.config()

    c.put('game', 'Pong-v0')

    c.put('state_steps', 4)

    c.put('summary_update_steps', 1)

    c.put('update_follower_steps', 10000)
    c.put('start_train_after_steps', 50000)

    c.put('q_alpha', 1.0)
    c.put('discount_gamma', 0.99)
    c.put('epsilon_start', 1.0)
    c.put('epsilon_end', 0.02)
    c.put('initial_explore_steps', 5000)
    c.put('total_explore_steps', 100000)

    c.put('learning_rate_start', 0.001)
    c.put('learning_rate_step', 100000)
    c.put('learning_rate_factor', 0.5)

    c.put('dense_layer_units', 512)

    c.put('input_shape', (80, 80))

    c.put('history_size', 100000)
    c.put('batch_size', 32)
    c.put('train_interval', 1)

    c.put('output_path', 'output')

    q = qlearn.qlearn(c)
    q.run(10000)
