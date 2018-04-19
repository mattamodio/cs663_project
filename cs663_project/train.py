from collections import namedtuple

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from skimage import color
from utils import now
from model import DiscoGAN
from loader import Loader
from data_tools import *


#batch1, batch2, channels, imdim = get_data_mnist(); data='mnist'; batch_size=10
batch1, batch2, channels, imdim = get_data_cifar(); data='cifar'; batch_size=10
#batch1, batch2, channels, imdim = get_data_monet(); data='monet'; batch_size=1
#batch1, batch2, channels, imdim = get_data_zebra(); data='zebra'; batch_size=1
SAVEFOLDER = 'model_{}/'.format(data);
ATTN = False

def main():
    load1 = Loader(batch1, shuffle=True)
    load2 = Loader(batch2, shuffle=True)

    discogan = DiscoGAN(dim_b1=imdim, dim_b2=imdim, channels=channels, attn=ATTN)

    if not os.path.exists(SAVEFOLDER): os.mkdir(SAVEFOLDER)
    plt.ioff(); fig = plt.figure(figsize=(6, 12))
    np.set_printoptions(precision=3)
    decay = discogan.learning_rate / 100.

    for i in range(1, 200000):
        if i % 10 == 0:
            print("Iter {} ({})".format(i, now()))

        xb1 = load1.next_batch(batch_size)
        xb2 = load2.next_batch(batch_size)

        xb1 = np.stack([randomize_image(
            img.reshape((imdim, imdim, channels)), enlarge_size=int(1.2 * imdim), output_size=imdim)
            for img in xb1], 0)
        xb2 = np.stack([randomize_image(
            img.reshape((imdim, imdim, channels)), enlarge_size=int(1.2 * imdim), output_size=imdim)
            for img in xb2], 0)

        discogan.train(xb1, xb2)

        if i >= 100000 and i % 1000 == 0:
            discogan.learning_rate -= decay
            print("Lowered learing rate to {:.5f}".format(discogan.learning_rate))

        if i and i % 500 == 0:
            testb1 = load1.next_batch(10)
            testb2 = load2.next_batch(10)
            print(discogan.get_loss_names())
            lstring = discogan.get_loss(xb1, xb2)
            print("{} ({}): {}".format(i, now(), lstring))

            xb1_reconstructed = discogan.get_layer(testb1, testb2, 'xb1_reconstructed')
            Gb1 = discogan.get_layer(testb1, testb2, 'Gb1')
            xb2_reconstructed = discogan.get_layer(testb1, testb2, 'xb2_reconstructed')
            Gb2 = discogan.get_layer(testb1, testb2, 'Gb2')
            if ATTN:
                attb1 = discogan.get_layer(testb1, testb2, 'attnb1')
                attb2 = discogan.get_layer(testb1, testb2, 'attnb2')


            testb1 = (testb1 + 1) / 2
            testb2 = (testb2 + 1) / 2
            xb1_reconstructed = (xb1_reconstructed + 1) / 2
            Gb1 = (Gb1 + 1) / 2
            xb2_reconstructed = (xb2_reconstructed + 1) / 2
            Gb2 = (Gb2 + 1) / 2

            print(Gb1.min(), Gb1.max(), xb1_reconstructed.min(), xb1_reconstructed.max())

            fig.clf()
            fig.subplots_adjust(.01, .01, .99, .99, .01, .01)
            for ii in range(10):
                if ATTN:
                    ax1 = fig.add_subplot(10, 4, 4 * ii + 1)
                    ax2 = fig.add_subplot(10, 4, 4 * ii + 2)
                    ax3 = fig.add_subplot(10, 4, 4 * ii + 3)
                    ax4 = fig.add_subplot(10, 4, 4 * ii + 4)
                    for ax in (ax1, ax2, ax3, ax4):
                        ax.set_xticks([])
                        ax.set_yticks([])
                else:
                    ax1 = fig.add_subplot(10, 3, 3 * ii + 1)
                    ax2 = fig.add_subplot(10, 3, 3 * ii + 2)
                    ax3 = fig.add_subplot(10, 3, 3 * ii + 3)
                    for ax in (ax1, ax2, ax3):
                        ax.set_xticks([])
                        ax.set_yticks([])
                if channels==1:
                    ax1.imshow(testb1[ii].reshape((imdim, imdim)), cmap='gray', vmin=0, vmax=1)
                    ax2.imshow(Gb2[ii].reshape((imdim, imdim)), cmap='gray', vmin=0, vmax=1)
                    ax3.imshow(xb1_reconstructed[ii].reshape((imdim, imdim)), cmap='gray', vmin=0, vmax=1)
                else:
                    ax1.imshow(testb1[ii].reshape((imdim, imdim, channels)))
                    ax2.imshow(Gb2[ii].reshape((imdim, imdim, channels)))
                    ax3.imshow(xb1_reconstructed[ii].reshape((imdim, imdim, channels)))
                    if ATTN:
                        x = attb1[ii]
                        x = x.reshape((imdim,imdim,3)); x = x.sum(axis=2)
                        img_hsv = color.rgb2hsv(testb1[ii].reshape((imdim, imdim, channels)))
                        x = x - x.min(); x = x / x.max();
                        img_hsv[:,:,2] *= x
                        img_masked = color.hsv2rgb(img_hsv)
                        ax4.imshow(img_masked)
            fig.canvas.draw()
            fig.savefig('b1_to_b2_{}.png'.format(data), dpi=500)


            fig.clf()
            fig.subplots_adjust(.01, .01, .99, .99, .01, .01)
            for ii in range(10):
                if ATTN:
                    ax1 = fig.add_subplot(10, 4, 4 * ii + 1)
                    ax2 = fig.add_subplot(10, 4, 4 * ii + 2)
                    ax3 = fig.add_subplot(10, 4, 4 * ii + 3)
                    ax4 = fig.add_subplot(10, 4, 4 * ii + 4)
                    for ax in (ax1, ax2, ax3, ax4):
                        ax.set_xticks([])
                        ax.set_yticks([])
                else:
                    ax1 = fig.add_subplot(10, 3, 3 * ii + 1)
                    ax2 = fig.add_subplot(10, 3, 3 * ii + 2)
                    ax3 = fig.add_subplot(10, 3, 3 * ii + 3)
                    for ax in (ax1, ax2, ax3):
                        ax.set_xticks([])
                        ax.set_yticks([])
                if channels==1:
                    ax1.imshow(testb2[ii].reshape((imdim, imdim)), cmap='gray', vmin=0, vmax=1)
                    ax2.imshow(Gb1[ii].reshape((imdim, imdim)), cmap='gray', vmin=0, vmax=1)
                    ax3.imshow(xb2_reconstructed[ii].reshape((imdim, imdim)), cmap='gray', vmin=0, vmax=1)
                else:
                    ax1.imshow(testb2[ii].reshape((imdim, imdim, channels)))
                    ax2.imshow(Gb1[ii].reshape((imdim, imdim, channels)))
                    ax3.imshow(xb2_reconstructed[ii].reshape((imdim, imdim, channels)))
                    if ATTN:
                        x = attb2[ii]
                        x = x.reshape((imdim,imdim,3)); x = x.sum(axis=2)
                        img_hsv = color.rgb2hsv(testb2[ii].reshape((imdim, imdim, channels)))
                        x = x - x.min(); x = x / x.max();
                        img_hsv[:,:,2] *= x
                        img_masked = color.hsv2rgb(img_hsv)
                        ax4.imshow(img_masked)
            fig.canvas.draw()
            fig.savefig('b2_to_b1_{}.png'.format(data), dpi=500)

            print('Saved.')
            discogan.save(folder=SAVEFOLDER)


if __name__ == '__main__':
    main()
