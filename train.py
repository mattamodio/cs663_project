import numpy as np
import glob, pickle, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imrotate, imresize
from skimage import color
import scipy.ndimage
from utils import now
from model import DiscoGAN
from loader import Loader


ATTN = False

def randomize_image(img, enlarge_size=286, output_size=256):
    img = imresize(img, [enlarge_size, enlarge_size])
    h1 = int(np.ceil(np.random.uniform(1e-2, enlarge_size-output_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, enlarge_size-output_size)))
    img = img[h1:h1+output_size, w1:w1+output_size]

    if np.random.random() > 0.5:
        img = np.fliplr(img)
    if np.random.random() > 0.5:
        img = np.flipud(img)

    img = img.reshape((-1))

    img = (img / 127.5) - 1

    return img

def get_data_cifar(dir='cifar/data_batch_*'):
    files_ = glob.glob(dir)
    data = []
    labels = []
    for file_ in files_:
        with open(file_, mode='rb') as f:
            d = pickle.load(f, encoding='bytes')
            data.append(d[b'data'])
            labels.append(d[b'labels'])

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    data = (data / 127.5) - 1 #data / 255.
    data = data.reshape((-1,3,32,32)).transpose(0,2,3,1).reshape((-1,32*32*3))

    batch1 = data[labels==0]
    batch2 = data[labels==1]

    return batch1, batch2, 3, 32

def get_data_mnist():
    """Return original and rotated MNIST."""
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    imgs = mnist.train.images.reshape((-1, 28, 28))
    #imgs = imgs[np.logical_or(mnist.train.labels == 3, mnist.train.labels == 7)]

    # batch 2
    imgs2 = imgs.copy()
    for i in range(imgs.shape[0]):
        imgs2[i] = imrotate(imgs2[i], 120) / 255.

    imgs = imgs.reshape((-1,28,28,1))
    imgs = np.pad(imgs, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

    imgs2 = imgs2.reshape((-1,28,28,1))
    imgs2 = np.pad(imgs2, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

    imgs = imgs.reshape((-1, 32*32))
    imgs2 = imgs2.reshape((-1, 32*32))

    return imgs, imgs2, 1, 32

def get_data_monet():
    b1 = glob.glob('/data/amodio/datasets/monet2photo/trainA/*')
    b2 = glob.glob('/data/amodio/datasets/monet2photo/trainB/*')

    b1 = np.stack([scipy.ndimage.imread(f) for f in b1], axis=0)
    b2 = np.stack([scipy.ndimage.imread(f) for f in b2], axis=0)

    b1 = b1.reshape((-1, 256*256*3))
    b2 = b2.reshape((-1, 256*256*3))

    b1 = b1 / 255.
    b2 = b2 / 255.

    return b1, b2, 3, 256

def get_data_zebra():
    b1 = glob.glob('/data/amodio/datasets/horse2zebra/trainA/*')
    b2 = glob.glob('/data/amodio/datasets/horse2zebra/trainB/*')
    b2 = [b for i,b in enumerate(b2) if i not in [136, 233, 287, 390, 813, 1160, 1227]] # some images are not 256x256x3...

    b1 = np.stack([scipy.ndimage.imread(f) for f in b1], axis=0)
    b2 = np.stack([scipy.ndimage.imread(f) for f in b2], axis=0)

    b1 = b1.reshape((-1, 256*256*3))
    b2 = b2.reshape((-1, 256*256*3))

    b1 = (b1 / 127.5) - 1# b1 / 255.
    b2 = (b2 / 127.5) - 1# b2 / 255.

    return b1, b2, 3, 256

#batch1, batch2, channels, imdim = get_data_mnist(); data='mnist'; batch_size=10
batch1, batch2, channels, imdim = get_data_cifar(); data='cifar'; batch_size=10
#batch1, batch2, channels, imdim = get_data_monet(); data='monet'; batch_size=1
#batch1, batch2, channels, imdim = get_data_zebra(); data='zebra'; batch_size=1


load1 = Loader(batch1, shuffle=True)
load2 = Loader(batch2, shuffle=True)

discogan = DiscoGAN(dim_b1=imdim, dim_b2=imdim, channels=channels, attn=ATTN)


SAVEFOLDER = 'model_{}/'.format(data);
if not os.path.exists(SAVEFOLDER): os.mkdir(SAVEFOLDER)
plt.ioff(); fig = plt.figure(figsize=(6, 12))
np.set_printoptions(precision=3)
decay = discogan.learning_rate / 100.

for i in range(1, 200000):
    if i%10==0: print("Iter {} ({})".format(i, now()))
    xb1 = load1.next_batch(batch_size)
    xb2 = load2.next_batch(batch_size)

    xb1 = np.stack([randomize_image(img.reshape((imdim,imdim,channels))) for img in xb1], 0)
    xb2 = np.stack([randomize_image(img.reshape((imdim,imdim,channels))) for img in xb2], 0)

    discogan.train(xb1, xb2)

    if i>=100000 and i%1000==0:
        discogan.learning_rate -= decay
        print("Lowered learing rate to {:.5f}".format(discogan.learning_rate))

    if i and i%500==0:
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


        testb1 = (testb1+1) / 2
        testb2 = (testb2+1) / 2
        xb1_reconstructed = (xb1_reconstructed+1) / 2
        Gb1 = (Gb1+1) / 2
        xb2_reconstructed = (xb2_reconstructed+1) / 2
        Gb2 = (Gb2+1) / 2
            
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











