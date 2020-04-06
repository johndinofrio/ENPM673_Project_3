from PIL import Image, ImageOps
# import cv2 as cv
from os import path
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def image_module(im):


        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        # im.thumbnail(new_size, Image.ANTIALIAS)

        im = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))

        # new_im.show()

        delta_w = desired_size - new_size[0]
        delta_h = desired_size - new_size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_im = ImageOps.expand(im, padding)
        print(new_im.size)
        # new_im.show()
        return new_im

def resize_cv(im):


    old_size = im.shape[:2]  # old_size is in (height, width) format
    print(old_size)
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,
                                value=color)
    print(new_im.shape[:2])
    # cv.imshow("image", new_im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return new_im



if __name__ == '__main__':
    hb = np.zeros((100, 100),dtype=np.uint8)
    hr = np.zeros((100, 100),dtype=np.uint8)
    hg = np.zeros((100, 100),dtype=np.uint8)
    desired_size = 100
    Folder=2
    if Folder==0:
        path = glob.glob("Orange_Trained/*.png")
    elif Folder ==1:
        path = glob.glob("Green_Trained/*.png")
    elif Folder==2:
        path = glob.glob("Yellow_Trained/*.png")

    for num, img in enumerate(path):
        im = cv.imread(img)
        # im1 = Image.open(img)
        # new = image_module(im1)
        new1 = resize_cv(im)
        if Folder == 2:
            cv.imwrite("Orange_Resized/{}.png".format(num), new1)
        elif Folder == 1:
            cv.imwrite("Green_Resized/{}.png".format(num), new1)
        elif Folder == 2:
            cv.imwrite("Yellow_Resized/{}.png".format(num), new1)
