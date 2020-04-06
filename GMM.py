import numpy as np


class GMM:
    def __init__(self, nClusters, epsilon=1e-5, maxIt=1000):
        self._data = None
        self._nPixels = None
        self._nFeatures = None

        self._nClusters = nClusters
        self._epsilon = epsilon
        self._maxIt = maxIt

        self._parameters = None
        self._probabilities = None

        self._mixture = 1 / self._nClusters * np.ones((1, self._nClusters))
        self._likelihoodLog = []

        self._likelihood = None
        self._classes = None

    def _initParameters(self):
        mu = self._data[np.random.choice(self._nPixels, 1)]
        cov = np.multiply(np.random.randint(1, 255) * np.eye(self._nFeatures), np.random.rand(self._nFeatures, self._nFeatures))

        return {'mu': mu, 'cov': cov}

    def _computeProbability(self, mu, cov):
        diff = np.subtract(self._data, mu)
        N = np.exp(-0.5 * np.sum(np.multiply(np.dot(diff, np.linalg.inv(cov)), diff), axis=1)) / ((2.0 * np.pi) ** (self._nFeatures / 2.0) * (np.linalg.det(cov) ** 0.5))

        return N.reshape(-1,)

    def _initTrain(self):
        self._parameters = [self._initParameters() for _ in range(self._nClusters)]
        self._probabilities = np.zeros((self._nPixels, self._nClusters))

    def _expectationStep(self):
        for i in range(self._nClusters):
            self._probabilities[:, i] = self._computeProbability(self._parameters[i]['mu'], self._parameters[i]['cov']) * self._mixture[0, i]

        sumCluster = np.sum(self._probabilities, axis=1)
        likelihood = np.sum(np.log(sumCluster))

        self._likelihoodLog.append(likelihood)
        self._probabilities = np.divide(self._probabilities, np.tile(sumCluster, (self._nClusters, 1)).transpose())

        return np.sum(self._probabilities, axis=0)

    def _maximizationStep(self, N):
        for i in range(self._nClusters):
            self._parameters[i]['mu'] = 1 / N[i] * np.sum(np.dot(self._probabilities[:, i], self._data))

            diff = np.subtract(self._data, self._parameters[i]['mu'])
            self._parameters[i]['cov'] = 1 / N[i] * np.dot(np.multiply(diff.transpose(), self._probabilities[:, i]), diff)

            self._mixture[0, i] = 1 / self._nPixels * N[i]

    def train(self, data, params=None, mixture=None):
        self._data = data
        self._nPixels = self._data.shape[0]  # 12000
        self._nFeatures = self._data.shape[1]  # 1

        if params is None and mixture is None:
            self._initTrain()
        else:
            self._parameters = params
            self._mixture = mixture
            self._probabilities = np.zeros((self._nPixels, self._nClusters))

        for j in range(self._maxIt):
            N = self._expectationStep()
            self._maximizationStep(N)

            if j > 0 and np.abs(self._likelihoodLog[-1] - self._likelihoodLog[-2]) < self._epsilon:
                break

        return self._parameters, self._mixture

    def _initTest(self):
        self._probabilities = np.zeros((self._nPixels, self._nClusters))
        self._classification = np.zeros((self._nPixels, self._nClusters))
        self._likelihood = np.zeros((self._nPixels, 1))
        self._classes = np.zeros((self._nPixels, 1))

    def test(self, data, parameters, mixture, thres=1.):
        self._data = data
        self._nPixels = self._data.shape[0]  # 12000
        self._nFeatures = self._data.shape[1]  # 1

        self._parameters = parameters
        self._mixture = mixture

        self._initTest()

        for i in range(self._nClusters):
            self._probabilities[:, i] = self._computeProbability(self._parameters[i]['mu'], self._parameters[i]['cov']) * self._mixture[0, i]
            self._classification[:, i] = np.where(self._probabilities[:, i] > np.max(self._probabilities[:, i]) / thres, i + 1, 0)

        self._likelihood = np.sum(self._probabilities, axis=1)
        self._classes = np.sum(self._classification, axis=1)

        return self._likelihood, self._classes
