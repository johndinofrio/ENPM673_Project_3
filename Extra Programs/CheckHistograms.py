from ComputeHistrograms import drawHistogram
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    pickle_in = open("../Perception_Project3/HistogramsData.pickle", "rb")
    fileData = pickle.load(pickle_in)

    # Color histogram
    fig1 = plt.figure(1)

    axs1 = fig1.add_subplot(131)
    axs2 = fig1.add_subplot(132)
    axs3 = fig1.add_subplot(133)

    axs1.grid(True)
    axs2.grid(True)
    axs3.grid(True)

    # Color combinations
    color = ['b', 'r', 'y']

    for i, key in enumerate(fileData.keys()):
        data = fileData[key]

        axs1.plot(data['red'])
        axs2.plot(data['green'])
        axs3.plot(data['blue'])

        drawHistogram(data['red'], data['green'], data['blue'], key)

    axs1.set_title('red')
    axs2.set_title('green')
    axs3.set_title('blue')

    fig1.tight_layout()
    plt.show()
