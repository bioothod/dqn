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

    c.put('game', 'PongNoFrameskip-v4')

    c.put('state_steps', 4)

    c.put('summary_update_steps', 1)

    c.put('update_follower_steps', 800)

    c.put('q_alpha', 1.0)
    c.put('discount_gamma', 0.99)

    c.put('learning_rate_start', 0.00025)
    c.put('learning_rate_step', 100000)
    c.put('learning_rate_factor', 0.7)

    c.put('dense_layer_units', 512)

    c.put('input_shape', (84, 84, 1))

    c.put('history_size', 10000)
    c.put('start_train_after_steps', 0)

    c.put('batch_size', 32)
    c.put('train_interval', 1)

    c.put('output_path', 'output')

    q = qlearn.qlearn(c)
    q.run(10000)
