import numpy as np

from collections import deque

import logging

logger = logging.getLogger(__name__)

class state(object):
    def __init__(self, shape, size):
        self.steps = deque()
        self.shape = shape
        self.size = size
        self.reshape = False
        self.value = None

        for i in range(size):
            self.push_zeroes()

    def push_zeroes(self):
        self.push_tensor(np.zeros(self.shape))

    def push_tensor(self, st):
        if self.shape != st.shape:
            logger.critical("self.shape: %s, tensor.shape: %s" % (self.shape, st.shape))
            assert self.shape == st.shape

        if len(self.steps) == self.size:
            self.steps.popleft()

        self.steps.append(st)
        self.reshape = True

    def merge(self):
        #self.value = np.concatenate(self.steps, axis=len(self.shape)-1)
        self.value = np.ndarray((self.shape[0], self.shape[1], self.size))
        for i, s in enumerate(self.steps):
            self.value[:,:,i] = s
        self.reshape = False

    def read(self):
        if self.reshape:
            self.merge()
        return self.value
