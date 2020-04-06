import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os


def ComputeDrawHistogram(img):
    color = ('b', 'g', 'r')

    fig, axs = plt.subplots(2, 2)

    for i, col in enumerate(color):
        colHist = cv2.calcHist([img], [i], None, [256], [1, 256])
        axs[0, 0].plot(colHist, color=col)
        axs[0, 0].set_xlim([0, 256])

    # imgHist = axs[0, 0].hist(img.ravel(), 256, [1, 256], facecolor='y')
    # axs[0, 0].set_xlim([0, 256])

    b, g, r = cv2.split(img)

    blueHist = axs[0, 1].hist(b.ravel(), 256, [1, 256], facecolor='b')
    axs[1, 1].set_xlim([0, 256])
    greenHist = axs[1, 0].hist(g.ravel(), 256, [1, 256], facecolor='g')
    axs[1, 0].set_xlim([0, 256])
    redHist = axs[1, 1].hist(r.ravel(), 256, [1, 256], facecolor='r')
    axs[1, 1].set_xlim([0, 256])

    return blueHist[0], greenHist[0], redHist[0]


def ComputeHistogram(img):
    blueHist = cv2.calcHist([img], [0], None, [256], [1, 256])
    greenHist = cv2.calcHist([img], [1], None, [256], [1, 256])
    redHist = cv2.calcHist([img], [2], None, [256], [1, 256])

    return blueHist, greenHist, redHist


def DrawHistogram(redHist, greenHist, blueHist, title=None):
    fig, axs = plt.subplots(1, 3)

    axs[0].plot(blueHist, color='b')
    axs[0].set_xlim([0, 256])
    axs[0].grid()

    axs[1].plot(greenHist, color='g')
    axs[1].set_xlim([0, 256])
    axs[1].grid()

    axs[2].plot(redHist, color='r')
    axs[2].set_xlim([0, 256])
    axs[2].grid()

    if title:
        plt.suptitle(title)

    return None


def AverageHistogram(path):
    files = os.listdir(path)

    sumBlueHist = 0
    sumGreenHist = 0
    sumRedHist = 0

    for file in files:
        filename = os.path.join(path, file)
        img = cv2.imread(filename)
        cv2.destroyWindow('img')
        cv2.imshow('img', img)

        blueHist, greenHist, redHist  = ComputeHistogram(img)

        sumBlueHist = np.add(sumBlueHist, blueHist)
        sumGreenHist = np.add(sumGreenHist, greenHist)
        sumRedHist = np.add(sumRedHist, redHist)

    avgBlueHist = sumBlueHist / len(files)
    avgGreenHist = sumGreenHist / len(files)
    avgRedHist = sumRedHist / len(files)

    DrawHistogram(avgRedHist, avgGreenHist, avgBlueHist)

    plt.show()

    return avgBlueHist, avgGreenHist, avgRedHist


if __name__ == '__main__':
    paths = ['Orange_Trained/', 'Green_Trained/', 'Yellow_Trained/']

    fileData = {}

    for path in paths:
        saveData = {'blue': None, 'green': None, 'red': None}

        blueHist, greenHist, redHist = AverageHistogram(path)

        saveData['blue'] = blueHist
        saveData['green'] = greenHist
        saveData['red'] = redHist

        fileData[path] = saveData

    # pickle_out = open("HistogramsData.pickle", "wb")
    # pickle.dump(fileData, pickle_out)
    # pickle_out.close()









