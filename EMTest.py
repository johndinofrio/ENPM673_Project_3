import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    # Draw pdf
    def pdf(self):
        x = np.linspace(int(self.sigma*(self.mu - 10)), int(self.sigma*(self.mu + 10)), 1000)

        # Normalization
        xnorm = (x - self.mu) / abs(self.sigma)

        # Probability estimation
        y = (1 / (np.sqrt(2 * np.pi) * abs(self.sigma))) * np.exp(-xnorm * xnorm / 2)

        # Draw PDF
        plt.plot(x, y)
        plt.grid()
        plt.show()

        return None

    def prob(self, x):
        # Data normalization
        xnorm = (x - self.mu) / abs(self.sigma)

        # Probability estimation
        return (1 / (np.sqrt(2 * np.pi) * abs(self.sigma))) * np.exp(-xnorm * xnorm / 2)

    # Return parameters
    def params(self):
        return self.mu[0], self.sigma


class GaussianMixture:
    def __init__(self, data, muMin, muMax, sigmaMin=0.1, sigmaMax=1, mixture=0.5):
        self.data = data

        # Gaussian distributions initializations
        self.gauss1 = Gaussian(np.random.uniform(muMin, muMax), np.random.uniform(sigmaMin, sigmaMax))
        self.gauss2 = Gaussian(np.random.uniform(muMin, muMax), np.random.uniform(sigmaMin, sigmaMax))
        self.gauss3 = Gaussian(np.random.uniform(muMin, muMax), np.random.uniform(sigmaMin, sigmaMax))

        # Mixture term initialization
        self.mixture = mixture

        self.loglike = 0

    def Expectation(self):
        # self.loglike = 0

        for d in self.data:
            # Non-normalized weights
            weight1 = self.gauss1.prob(d) * self.mixture
            weight2 = self.gauss2.prob(d) * (1 - self.mixture)

            # Compute denominator
            den = weight1 + weight2

            # Normalize weights
            weight1 /= den
            weight2 /= den

            # Add loglike
            self.loglike += np.log(weight1 + weight2)

            yield (weight1, weight2)

    def Maximization(self, weights):
        # Compute denominators
        (left, right) = zip(*weights)

        gauss1Den = sum(left)
        gauss2Den = sum(right)

        # Compute updated means
        self.gauss1.mu = sum(w * d / gauss1Den for (w, d) in zip(left, self.data))
        self.gauss2.mu = sum(w * d / gauss2Den for (w, d) in zip(right, self.data))

        # Compute updated sigmas
        self.gauss1.sigma = np.sqrt(sum(w * ((d - self.gauss1.mu) ** 2) for (w, d) in zip(left, self.data)) / gauss1Den)
        self.gauss2.sigma = np.sqrt(sum(w * ((d - self.gauss2.mu) ** 2) for (w, d) in zip(right, self.data)) / gauss2Den)

        # Compute updated mixture
        self.mixture = gauss1Den / len(self.data)

    def iterate(self):
        weights = self.Expectation()
        self.Maximization(weights)

        self.printVerbose()

    # def pdf(self, x):
    #     return self.mixture * self.gauss1.pdf(x) + (1 - self.mixture) * self.gauss2.pdf(x)

    def params(self):
        return self.gauss1.params(), self.gauss2.params(), self.mixture

    def printVerbose(self):
        print('Mixture: {}, {}, mix={}'.format(self.gauss1.params(), self.gauss2.params(), self.mixture))


if __name__ == '__main__':
    # gauss1 = Gaussian(0, 0.25)
    # gauss1.pdf()

    data1 = np.random.normal(2, 1, 4000)
    data2 = np.random.normal(10, 2, 8000)

    data = np.hstack([data1, data2]).reshape(-1, 1)

    n_iterations = 20
    model = GaussianMixture(data, min(data), max(data))

    for _ in range(n_iterations):
        model.iterate()
        mix = model.params()

    print(mix)

    gauss1 = Gaussian(mix[0][0], mix[0][1])
    gauss2 = Gaussian(mix[1][0], mix[1][1])
    mixture = mix[2]

    x = np.linspace(0, 20, 1000)
    y = (mixture * gauss1.prob(x) + (1 - mixture) * gauss2.prob(x))

    plt.plot(x, y)
    sns.distplot(data, bins=20, kde=False, norm_hist=True)
    plt.show()



