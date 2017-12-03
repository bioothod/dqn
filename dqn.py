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

    c.put('num_atoms', 51)
    c.put('v_min', -10.)
    c.put('v_max', +10.)

    c.put('state_steps', 4)

    c.put('summary_update_steps', 100)

    c.put('update_follower_steps', 1000)

    c.put('q_alpha', 1.0)
    c.put('discount_gamma', 0.99)
    c.put('epsilon_start', 1.0)
    c.put('epsilon_end', 0.01)
    c.put('initial_explore_steps', 0)
    c.put('total_explore_steps', 150000)

    c.put('learning_rate_start', 2.5e-4)
    c.put('learning_rate_step', 100000)
    c.put('learning_rate_factor', 0.7)

    c.put('dense_layer_units', 512)

    c.put('input_shape', (84, 84, 1))

    c.put('history_size', 30000)
    c.put('start_train_after_steps', 0)

    c.put('batch_size', 32)
    c.put('train_interval', 1)

    c.put('output_path', 'output')

    q = qlearn.qlearn(c)
    q.run(250)
