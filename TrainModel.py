from tqdm import tqdm
from GMM import GMM
import numpy as np
import pickle
import cv2
import os


def loadImg(filename):
    img = cv2.imread(filename)
    cv2.imshow('flako',img)
    cv2.waitKey(1000)
    h, w, c = img.shape

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGrayReshaped = imgGray.reshape(h*w, -1)
    toDelete = np.where(imgGrayReshaped == 0)
    length = h*w - len(toDelete[0])
    imgArray = np.zeros((length, c)) ##TODO np.zeros((length, c))

    b, g, r = cv2.split(img)

    for i, channel in enumerate([b, g, r]):
        reshaped = channel.reshape(h*w, 1)
        cutted = np.delete(reshaped, toDelete)

        imgArray[:, i] = cutted

    return imgArray


if __name__ == '__main__':
    folder = 'models/'
    names = ['greenModel.pickle', 'orangeModel.pickle', 'yellowModel.pickle']
    paths = ['Green_Resized/', 'Orange_Resized/', 'Yellow_Resized/']
    gmms = [GMM(nClusters=3), GMM(nClusters=3), GMM(nClusters=3)] #TODO

    k = 2
    for path, gmm, name in zip(paths[k:3], gmms[k:3], names[k:3]): #TODO
        print(path)
        model = {}

        files = os.listdir(path)

        parameters, weights = None, None

        for file in tqdm(files):
            filename = os.path.join(path, file)
            imgArray = loadImg(filename)
            parameters, weights = gmm.train(imgArray, parameters, weights)

        model[path] = [parameters, weights]

        modelName = os.path.join(folder, name)

        pickle_out = open(modelName, "wb")
        pickle.dump(model, pickle_out)
        pickle_out.close()
