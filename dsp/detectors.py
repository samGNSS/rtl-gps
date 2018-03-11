import numpy as np

class CaCfar():
    """Cell averaging constant false alarm rate detector
    """
    def __init__(self, numGuardBins, numAvgBins, scale=0):
        self.back = np.arange(-numAvgBins, 0, 1)
        self.front = np.arange(numAvgBins)
        self.numGuardBins = numGuardBins
        self.numAvgBins = numAvgBins
        self.scale = scale

        self.forwardCells = None
        self.reverseCells = None
        self.forwardMean = 0
        self.reverseMean = 0

    def _initCells(self, data):
        """Initalize the averaging cells
        
        Arguments:
            data {np.array} -- 1d array of floats
        """

        self.forwardCells = (self.front + self.numGuardBins) % len(data)
        self.reverseCells = (self.back - self.numGuardBins) % len(data)

        self.forwardMean = data[self.forwardCells].sum()
        self.reverseMean = data[self.reverseCells].sum()

    def process(self, data):
        """Process the data

        [description]

        Arguments:
            data {np.array} -- 1d array of floats

        Returns:
            threshold and detection indices -- [description]
        """

        # init
        thresh = np.zeros_like(data)
        dets = np.zeros_like(data, dtype=bool)
        numPoints = len(data)

        self._initCells(data)
        # loop through the data
        for idx, cut in enumerate(data):
            # calculate threshold
            thresh[idx] = (self.forwardMean + self.reverseMean)/(2*self.numAvgBins) + self.scale

            # Check if we need to declare a detection
            dets[idx] = cut > thresh[idx]

            self.forwardMean -= data[self.forwardCells[0]]
            self.reverseMean -= data[self.reverseCells[0]]

            self.forwardCells += 1
            self.forwardCells %= numPoints

            self.reverseCells += 1
            self.reverseCells %= numPoints

            self.forwardMean += data[self.forwardCells[-1]]
            self.reverseMean += data[self.reverseCells[-1]]

        return thresh, dets


class OsCfar():
    """Ordered statistic constant false alarm rate detector
    """
    def __init__(self, numGuardBins, numAvgBins, osElement, scale=0):
        self.back = np.arange(-numAvgBins, 0, 1)
        self.front = np.arange(numAvgBins)
        self.numGuardBins = numGuardBins
        self.numAvgBins = numAvgBins
        self.osElement = osElement
        self.scale = scale

        self.forwardCells = None
        self.reverseCells = None
        self.cells = None

    def _initCells(self, data):
        self.forwardCells = (self.front + self.numGuardBins) % len(data)
        self.reverseCells = (self.back - self.numGuardBins) % len(data)

        self.cells = np.sort(np.array([data[self.forwardCells], data[self.reverseCells]]).flatten())

    def process(self, data):
        """[summary]

        [description]

        Arguments:
            data {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        # init
        thresh = np.zeros_like(data)
        dets = np.zeros_like(data, dtype=bool)
        numPoints = len(data)

        self._initCells(data)
        # loop through the data
        for idx, cut in enumerate(data):
            # calculate threshold
            thresh[idx] = self.cells[self.osElement] + self.scale

            # Check if we need to declare a detection
            dets[idx] = cut > thresh[idx]

            self.forwardCells += 1
            self.forwardCells %= numPoints

            self.reverseCells += 1
            self.reverseCells %= numPoints

            self.cells = np.sort(np.array([data[self.forwardCells], data[self.reverseCells]]).flatten())

        return thresh, dets

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    print("Running CFAR unit test")

    dataIn = np.exp(1j * 2*np.pi*1000*np.arange(10000) / 100e3)
    dataFFT = -100 + 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(dataIn, 1024))))

    # run the data through the cfar
    cfar = OsCfar(5, 10, int(10*0.75), 10)
    start = time.time()
    caThresh, caDets = cfar.process(dataFFT)
    endTime = time.time()

    print("Time to process {} samples: {}".format(len(dataFFT), endTime - start))

    # plot
    plt.figure()
    plt.plot(dataFFT)
    plt.plot(caThresh)
    plt.plot(np.arange(1024)[caDets], dataFFT[caDets], 'r*')
    plt.xlabel("FFT Bins")
    plt.ylabel("Power")
    plt.legend(["FFT", "Threshold"])
    plt.grid()
    plt.show()