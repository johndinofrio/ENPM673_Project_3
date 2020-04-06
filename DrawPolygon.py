import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
import numpy as np
import cv2
import os


def CleanCanvas(event):
    for p in pointPlot:
        p.remove()

    # originImg.set_data(img)
    croppedImg.set_data(imgEmpty)
    maskedImg.set_data(imgEmpty)

    points.clear()
    pointPlot.clear()

    fig.canvas.draw()


def DrawPoint(event):
    global maskedGlobal

    if event.button == 1:
        x, y = int(event.xdata), int(event.ydata)

        if x != 0 and y != 0:
            points.append([x, y])
            print("x={}, y={}".format(x, y))
            pointPlot.extend(ax1.plot(x, y, 'wo', markersize=1))

            if len(points) > 2:
                pointsAsArray = np.array(points)
                x, y, w, h = cv2.boundingRect(pointsAsArray)
                cropped = img[y:y + h, x:x + w].copy()
                croppedImg.set_data(cropped)

                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.fillPoly(mask, np.array([pointsAsArray]), 255)
                res = cv2.bitwise_and(img, img, mask=mask)
                masked = res[y:y + h, x:x + w].copy()
                maskedImg.set_data(masked)

                maskedGlobal = masked

            fig.canvas.draw()


def SaveFrame(event):
    name = 'Green_{}.png'.format(format(frameCont, '03'))
    plt.imsave(os.path.join(savingPath, name), maskedGlobal)

    for p in pointPlot:
        p.remove()

    # originImg.set_data(img)
    croppedImg.set_data(imgEmpty)
    maskedImg.set_data(imgEmpty)

    points.clear()
    pointPlot.clear()

    fig.canvas.draw()

    plt.close(fig)


def ExitApp(event):
    global exitFlag

    exitFlag = True
    plt.close(fig)


if __name__ == '__main__':
    savingPath = 'Green_Trained'

    points = []
    pointPlot = []

    frameCont = 0
    exitFlag = False

    maskedGlobal = None

    cam = cv2.VideoCapture('./detectbuoy.avi')
    cont=0
    while cam.isOpened():

        ret, frame = cam.read()

        if int(cont%2)==0:
            print(cont)

            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                fig = plt.figure()
                fig.subplots_adjust(bottom=0.2)

                gs = GridSpec(nrows=2, ncols=2)

                mng = plt.get_current_fig_manager()
                # mng.window.showMaximized()

                axclean = plt.axes([0.1, 0.05, 0.1, 0.075])
                bclean = Button(axclean, 'Clean')
                bclean.on_clicked(CleanCanvas)

                axsave = plt.axes([0.21, 0.05, 0.1, 0.075])
                bsave = Button(axsave, 'Save')
                bsave.on_clicked(SaveFrame)

                axexit = plt.axes([0.32, 0.05, 0.1, 0.075])
                bexit = Button(axexit, 'Exit')
                bexit.on_clicked(ExitApp)

                fig.canvas.mpl_connect('button_press_event', DrawPoint)

                ax1 = fig.add_subplot(gs[:, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, 1])

                ax1.axis('off')
                ax2.axis('off')
                ax3.axis('off')

                ax1.set_title('Frame: {}'.format(frameCont))
                frameCont += 1

                # img = plt.imread('imgTest.PNG')
                imgEmpty = 255*np.ones_like(img)

                originImg = ax1.imshow(img)
                croppedImg = ax2.imshow(imgEmpty)
                maskedImg = ax3.imshow(imgEmpty)

                plt.tight_layout()
                plt.show()

            if exitFlag:
               break
        cont += 1
