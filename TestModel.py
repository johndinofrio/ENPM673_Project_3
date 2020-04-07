from GMM import GMM
import numpy as np
import pickle
import cv2 as cv
import os
import matplotlib.pyplot as plt


def computeMask(img, model, params, mixture, thres=1):
    h, w, c = img.shape
    b, g, r = cv.split(img)
    data = np.zeros((h * w, 3))  # TODO

    for i, channel in enumerate([b, g, r]):  # TODO
        data[:, i] = channel.reshape(h * w, )

    probs, _ = model.test(data, params, mixture)
    probs[probs > np.max(probs) / thres] = 255

    probsReshaped = probs.reshape((h, w))
    mask = np.zeros_like(img)

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
            if count>0:
                # Display the resulting frame
                # cv.imshow('Frame', img)
                # img = cv.cvtColor(img, cv.COLOR_BGRA2RGB)
                # plt.hist(img, 255)
                # plt.show()

                # Morphology 
                yMask = computeMask(img, yModel, yParams, yMixture, yThres)  # TODO
                yMask = cv.erode(yMask, kernel1, iterations=1)
                yMask = cv.dilate(yMask, kernel, iterations=1)
                yMask = cv.erode(yMask, kernel1, iterations=2)
                yMask = cv.dilate(yMask, kernel, iterations=1)
                yMask = cv.dilate(yMask, kernel2, iterations=2)
                yMask = cv.cvtColor(yMask, cv.COLOR_BGR2GRAY)
                if count<greenFrame:
                    gMask = computeMask(img, gModel, gParams, gMixture, gThres)  # TODO
                    gMask = cv.erode(gMask, kernel1, iterations=1)
                    gMask = cv.dilate(gMask, kernel, iterations=2)
                    gMask = cv.cvtColor(gMask, cv.COLOR_BGR2GRAY)
                if count<orangeFrame:
                    oMask = computeMask(img, oModel, oParams, oMixture, oThres)  # TODO           
                    oMask = cv.erode(oMask, kernel1, iterations=1)
                    oMask = cv.dilate(oMask, kernel, iterations=2)
                    oMask = cv.cvtColor(oMask, cv.COLOR_BGR2GRAY)
                if count>orangeFrame2:
                    oMask = computeMask(img, oModel, oParams, oMixture, oThres)  # TODO           
                    oMask = cv.erode(oMask, kernel1, iterations=1)
                    oMask = cv.dilate(oMask, kernel, iterations=4)
                    oMask = cv.cvtColor(oMask, cv.COLOR_BGR2GRAY)
                    yMask = computeMask(img, yModel, yParams, yMixture, yThres)  # TODO
                    yMask = cv.erode(yMask, kernel1, iterations=1)
                    yMask = cv.dilate(yMask, kernel2, iterations=5)
                    yMask = cv.cvtColor(yMask, cv.COLOR_BGR2GRAY)
                
                
                    # show Result
                    All = np.hstack([gMask, oMask])
                    cv.imshow('Frame1', All)
                    cv.imshow('Frame2', yMask)
                
                
                
                # draw Contours
                if count<orangeFrame or count>orangeFrame2:
                    # Orange
                    contours, hierarchy = cv.findContours(oMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    areas = [cv.contourArea(c) for c in contours]
                    max_index = np.argmax(areas)
                    cnt = contours[max_index]
                    cv.drawContours(img, [cnt], -1, (0, 127, 255), 5)

                # Yellow
                contours, hierarchy = cv.findContours(yMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                areas1 = [cv.contourArea(c) for c in contours]
                max_index1 = np.argmax(areas1)
                cnt1 = contours[max_index1]
                cv.drawContours(img, [cnt1], -1, (0, 255, 255), 5)

                if count<greenFrame:
                    # Green
                    contours, hierarchy = cv.findContours(gMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    areas2 = [cv.contourArea(c) for c in contours]
                    max_index2 = np.argmax(areas2)
                    cnt2 = contours[max_index2]
                    cv.drawContours(img, [cnt2], -1, (0, 128, 0), 5)
                
                # # calculate moments of binary image
                # M = cv.moments(gMask)

                # # calculate x,y coordinate of center
                # cX = int(M["m10"] / M["m00"])
                # cY = int(M["m01"] / M["m00"])
                # #
                # put text and highlight the center
                # cv.circle(img, (cX, cY), 10, (0, 128, 0), 2)
                # cv.drawContours(img, [cnt], -1, (0, 255, 0), 3)
                cv.imshow('Frame', img)
                # cv.waitKey(10)
                # Press Q on keyboard to  exit
                
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            count += 1
            print(count)
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()

    # TODO Combine masks to get the final solution
