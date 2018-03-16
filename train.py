import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imrotate
from utils import now
from model import DiscoGAN
from loader import Loader

PLOT = True
if PLOT:
    plt.ion()
np.set_printoptions(precision=3)



def get_data_mnist():
    """Return original and rotated MNIST."""
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    imgs = mnist.train.images.reshape((-1, 28, 28))
    imgs = imgs[np.logical_or(mnist.train.labels == 3, mnist.train.labels == 7)]

    # batch 2
    imgs2 = imgs.copy()
    for i in range(imgs.shape[0]):
        imgs2[i] = imrotate(imgs2[i], 120) / 255.

    imgs = imgs.reshape((-1, 28 * 28))
    imgs2 = imgs2.reshape((-1, 28 * 28))

    return imgs, imgs2

batch1, batch2 = get_data_mnist()

load1 = Loader(batch1, shuffle=True)
load2 = Loader(batch2, shuffle=True)

discogan = DiscoGAN(dim_b1=batch1.shape[1], dim_b2=batch2.shape[1])

if PLOT:
    fig1 = plt.figure(figsize=(8, 8))
    # fig2 = plt.figure(figsize=(8, 8))

for i in range(1, 100000):
    xb1 = load1.next_batch(250)
    xb2 = load2.next_batch(250)
    discogan.train(xb1, xb2)

    if i % 1000 == 0:
        print(discogan.get_loss_names())
        lstring = discogan.get_loss(xb1, xb2)
        print("{} ({}): {}".format(i, now(), lstring))

        if PLOT:
            xb1_reconstructed = discogan.get_layer(xb1, xb2, 'xb1_reconstructed')
            Gb1 = discogan.get_layer(xb1, xb2, 'Gb1')
            xb2_reconstructed = discogan.get_layer(xb1, xb2, 'xb2_reconstructed')
            Gb2 = discogan.get_layer(xb1, xb2, 'Gb2')

            print(Gb1.min(), Gb1.max(), xb1_reconstructed.min(), xb1_reconstructed.max())

            fig1.clf()
            fig1.subplots_adjust(.01, .01, .99, .99, .01, .01)
            for ii in range(10):
                ax1 = fig1.add_subplot(10, 3, 3 * ii + 1)
                ax2 = fig1.add_subplot(10, 3, 3 * ii + 2)
                ax3 = fig1.add_subplot(10, 3, 3 * ii + 3)
                for ax in (ax1, ax2, ax3):
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax1.imshow(xb1[ii].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
                ax2.imshow(Gb2[ii].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
                ax3.imshow(xb1_reconstructed[ii].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
            fig1.canvas.draw()


            # fig2.clf()
            # fig2.subplots_adjust(.01, .01, .99, .99, .01, .01)
            # for ii in range(10):
            #     ax1 = fig2.add_subplot(10, 3, 3 * ii + 1)
            #     ax2 = fig2.add_subplot(10, 3, 3 * ii + 2)
            #     ax3 = fig2.add_subplot(10, 3, 3 * ii + 3)
            #     for ax in (ax1, ax2, ax3):
            #         ax.set_xticks([])
            #         ax.set_yticks([])
            #     ax1.imshow(xb2[ii].reshape((28, 28)), cmap='gray', vmin=0)
            #     ax2.imshow(Gb1[ii].reshape((28, 28)), cmap='gray', vmin=0)
            #     ax3.imshow(xb2_reconstructed[ii].reshape((28, 28)), cmap='gray', vmin=0)
            # fig2.canvas.draw()

    
















