import random

class history(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.history = [None] * self.max_size

        # monotonically increasing counters
        # inserting element increases @end, removing element from the beginning increases @start
        self.start = 0
        self.end = 0

    def clear(self):
        self.__init__()

    def size(self):
        return self.end - self.start

    def full(self):
        return self.size() >= self.max_size

    def append(self, e):
        if self.start + self.max_size == self.end:
            self.start += 1

        offset = self.end % self.max_size
        self.history[offset] = e
        self.end += 1

    def get(self, idx):
        return self.history[(self.start + idx) % self.max_size]

    def sample(self, size):
        return random.sample(self.history, min(size, self.size()))

    def whole(self):
        ret = [0] * self.size()
        for i in range(len(ret)):
            ret[i] = self.get(i)

        return ret
