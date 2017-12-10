
class iqStream():
    def __init__(self, sdrHandle):
        self.sdr = sdrHandle

    @property
    def freq(self):
        return self.sdr.center_freq

    @freq.setter
    def freq(self, newFreq):
        self.sdr.center_freq = newFreq

    @property
    def rate(self):
        return self.sdr.sample_rate

    @rate.setter
    def rate(self, newRate):
        self.sdr.sample_rate = newRate

    def sample(self, numSamps):
        return self.sdr.read_samples(numSamps)
