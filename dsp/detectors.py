import numpy as np

"""
Cell averaging constant false alarm rate detector
"""
class CA_CFAR():
    def __init__(self, numGuardBins, numAvgBins, scale = 0):
        self.back = np.arange(-numAvgBins, 0, 1)
        self.front = np.arange(numAvgBins)
        self.numGuardBins = numGuardBins
        self.scale = scale

    def process(self, dataIn):
        threshTmp = np.zeros_like(dataIn)
        detsTmp = np.zeros_like(dataIn, dtype=bool)
        numPoints = len(dataIn)
        # loop throught the data
        for idx, cut in enumerate(dataIn):
            frontEstimate = dataIn[(idx + self.front + self.numGuardBins) % numPoints]
            backEstimate  = dataIn[(idx + self.back - self.numGuardBins) % numPoints]

            threshTmp[idx] = (frontEstimate.mean() + backEstimate.mean())/2 + self.scale

            if cut > threshTmp[idx]:
                detsTmp[idx] = True
            else:
                detsTmp[idx] = False


        return threshTmp, detsTmp

def CA_CFAR_NORM():
    pass

"""
Cell averaging cfar in 2 dimensions
"""
def CA_CFAR_2D():
    pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    print("Running CFAR unit test")

    data = np.exp(1j * 2*np.pi*1000*np.arange(10000) / 100e3)
    dataFFT = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data, 1024))))

    # run the data through the cfar
    cfar = CA_CFAR(5, 10, 5)
    start = time.time()
    thresh, dets = cfar.process(np.abs(dataFFT))
    endTime = time.time()

    print("Time to process {} samples: {}".format(len(dataFFT), endTime - start))

    plt.figure()
    plt.plot(dataFFT)
    plt.plot(thresh)
    plt.plot(np.arange(1024)[dets],dataFFT[dets],'r*')
    plt.show()


