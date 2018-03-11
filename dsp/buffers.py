import numpy as np

class CircularBuffer():
    """Simple circular buffer container
    """
    def __init__(self, length):
        self._validIdx = 0
        self._length = length
        self._wrap = False
        self._mem = np.empty(length)

    def append(self, data):
        self._mem[self._validIdx] = data
        self._validIdx += 1
        if self._validIdx == self._length:
            self._wrap = True
            self._validIdx = 0

    def appendList(self, vals):
        for val in vals:
            self.append(val)

    def get(self, idx):
        if idx > self._validIdx:
            return None
        else:
            return self._mem[idx]
    
    def getAll(self):
        if self._wrap:
            return self._mem
        else:
            return self._mem[:self._validIdx]


class Fifo():
    def __init__(self):
        pass


class MovingAverage(CircularBuffer):
    def __init__(self, numAverages):
        pass