import matplotlib
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from os import path
import glob
import cv2 as cv
Folder=1
if Folder == 0:
   path = glob.glob("Orange_Resized/*.png")
elif Folder == 1:
   path = glob.glob("Green_Resized/*.png")
elif Folder == 2:
   path = glob.glob("Yellow_Resized/*.png")

image_sum = np.zeros((100,100,3))

for image in path:
   im = cv.imread(image)
   # print(type(im))
   # cv.imshow('original', im)
   # cv.waitKey(0)
   # cv.destroyAllWindows()
   image_sum = image_sum.astype('uint8')
   # add=cv.addWeighted(image_sum, 0.2, im, 1, 0) # Yellow
   add = cv.addWeighted(image_sum, 0.3, im, 1, 0) # Green
   # add = cv.addWeighted(image_sum, 0.2, im, 0.9, 0)# red
   image_sum=np.copy(add)

cv.imshow('result', image_sum)
if Folder == 0:
   cv.imwrite("Orange_sum.png", image_sum)
elif Folder == 1:
   cv.imwrite("Green_sum.png", image_sum)
elif Folder == 2:
   cv.imwrite("Yellow_sum.png", image_sum)
image_sum = cv.cvtColor(image_sum, cv.COLOR_BGRA2RGB)
blueHist = cv.calcHist([image_sum], [0], None, [256], [1, 256])
greenHist = cv.calcHist([image_sum], [1], None, [256], [1, 256])
redHist = cv.calcHist([image_sum], [2], None, [256], [1, 256])
blueHist[250:255,0]=0
greenHist[250:255,0]=0
redHist[250:255,0]=0
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
fig=plt.figure()
r, g, b = cv.split(image_sum)
r = cv.cvtColor(r, cv.COLOR_BGRA2RGB)
g = cv.cvtColor(g, cv.COLOR_BGRA2RGB)
b= cv.cvtColor(b, cv.COLOR_BGRA2RGB)
fig.add_subplot(1,3, 1)
plt.imshow(b)
fig.add_subplot(1,3, 2)
plt.imshow(g)
fig.add_subplot(1,3, 3)
plt.imshow(r)
plt.show()
# image_sum = image_sum.astype('uint8')


cv.waitKey(0)
cv.destroyAllWindows()
