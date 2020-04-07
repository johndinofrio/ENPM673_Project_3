import matplotlib.pyplot as plt
from GMM import GMM
import numpy as np
import pickle
import cv2 as cv
import os

def computeMask(img, model, params, mixture, thres=1):
    h, w, c = img.shape
    b, g, r = cv.split(img)
    data = np.zeros((h * w, 3))  # TODO

    for i, channel in enumerate([b, g, r]):  # TODO
        data[:, i] = channel.reshape(h * w, )

    probs, classes = model.test(data, params, mixture)
    probs[probs > np.max(probs) / thres] = 255

    probsReshaped = probs.reshape((h, w))
    mask = np.zeros_like(img)
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
    x = np.linspace(np.min(data), np.max(data), 307200).reshape(-1, 1)

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
    mask[:, :, 0] = probsReshaped
    mask[:, :, 1] = probsReshaped
    mask[:, :, 2] = probsReshaped

    return mask
if __name__ == '__main__':
    # Loading models
    path = 'models/'
    files = os.listdir(path)

    paths = ['Green_Resized/', 'Orange_Resized/', 'Yellow_Resized/']
    gmms = [GMM(nClusters=3), GMM(nClusters=3), GMM(nClusters=3)]  # TODO

    gParams, gMixture = None, None
    oParams, oMixture = None, None
    yParams, yMixture = None, None

    for file in files:
        filename = os.path.join(path, file)

        pickle_in = open(filename, "rb")
        model = pickle.load(pickle_in)

        for key, info in model.items():
            if key == paths[0]:
                gParams = info[0]
                gMixture = info[1]
            elif key == paths[1]:
                oParams = info[0]
                oMixture = info[1]
            elif key == paths[2]:
                yParams = info[0]
                yMixture = info[1]
        print(gParams)
    gModel = gmms[0]
    oModel = gmms[1]
    yModel = gmms[2]

    # Select Threshold
    gThres, oThres, yThres = 3, 4, 2

    cap = cv.VideoCapture('detectbuoy.avi')
    kernel1 = np.ones((2, 2), np.uint8)
    kernel = np.array([[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 2, 2, 2, 1, 0],
                       [1, 2, 3, 3, 3, 2, 1],
                       [1, 2, 3, 4, 3, 2, 1],
                       [1, 2, 3, 3, 3, 2, 1],
                       [0, 1, 2, 2, 2, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]], dtype=np.uint8)
    kernel2 = np.ones((7, 3), np.uint8)
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Read until video is completed
    count = 0
    greenFrame = 46
    orangeFrame = 150
    orangeFrame2 = 170
    while cap.isOpened():
        # Capture frame-by-frame
        ret, img = cap.read()

        if ret == True:
            if count > 0:
                # Morphology
                yMask = computeMask(img, yModel, yParams, yMixture, yThres)  # TODO
                oMask = computeMask(img, oModel, oParams, oMixture, oThres)
                gMask = computeMask(img, gModel, gParams, gMixture, gThres)  # TODO
                cv.imshow('Frame', img)

                if cv.waitKey(100) & 0xFF == ord('q'):
                    break
            count += 1
            print(count)

        # Break the loop
        else:
            break


