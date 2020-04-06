import matplotlib.pyplot as plt
import numpy as np
from GMM import GMM

if __name__ == '__main__':
    data1, data2 = np.random.normal(2, 4, 4000), np.random.normal(20, 2, 8000)
    data = np.hstack([data1, data2]).reshape(12000, 1)

    gmm = GMM(nClusters=2)
    params, mix = gmm.train(data)
    # print(params, mix)

    probs, classes = gmm.test(data, params, mix, 10)
    # print(probs, classes)

    # GRAPHS
    fig = plt.figure()

    # Histogram
    ax = fig.add_subplot(311)
    ax.hist(data, 20, alpha=0.5)
    ax.set_title('Histogram')
    ax.set_xlabel('Data Range')
    ax.set_ylabel('Frequency')
    ax.grid(True)

    # Probability density function
    x = np.linspace(np.min(data), np.max(data), 1000).reshape(-1, 1)
    probs, _ = gmm.test(x, params, mix, 1.2)

    ax = fig.add_subplot(312)
    ax.plot(x, probs)
    ax.set_title('Probability Density Function')
    ax.set_xlabel('Data Range')
    ax.set_ylabel('Probability')
    ax.grid(True)

    # Classification
    ax = fig.add_subplot(313)
    ax.set_title('Data Classified')
    ax.set_xlabel('Data Range')
    ax.grid(True)

    xArray, yArray, cArray = [], [], []

    for i in range(len(classes)):
        x, y, c = data[i, 0], 0, classes[i]

        xArray.append(x)
        yArray.append(y)
        cArray.append(c)

    scatter = ax.scatter(xArray, yArray, c=cArray)
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)

    plt.tight_layout()
    plt.show()

