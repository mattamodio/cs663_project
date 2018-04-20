import glob
import os
import pickle
import re
import shutil
import zipfile

import numpy as np
import scipy.ndimage
from scipy.misc import imrotate, imresize

import cs663_project.common as common


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


def get_data(data_str):
    if data_str == 'cifar':
        return get_data_cifar()
    elif data_str == 'mnist':
        return get_data_mnist()
    elif data_str == 'monet':
        return get_data_monet()
    elif data_str == 'zebra':
        return get_data_zebra()
    else:
        raise ValueError("Unrecognized data set: {}".format(data_str))


def get_data_cifar(dir='data/cifar/data_batch_*'):
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

    data = (data / 127.5) - 1 # data / 255.
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
    imgs = np.pad(
        imgs, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

    imgs2 = imgs2.reshape((-1,28,28,1))
    imgs2 = np.pad(
        imgs2, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

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
    # some images are not 256x256x3...
    b2 = [b for i,b in enumerate(b2) if i not in [136, 233, 287, 390, 813, 1160, 1227]]

    b1 = np.stack([scipy.ndimage.imread(f) for f in b1], axis=0)
    b2 = np.stack([scipy.ndimage.imread(f) for f in b2], axis=0)

    b1 = b1.reshape((-1, 256*256*3))
    b2 = b2.reshape((-1, 256*256*3))

    b1 = (b1 / 127.5) - 1# b1 / 255.
    b2 = (b2 / 127.5) - 1# b2 / 255.

    return b1, b2, 3, 256


def get_data_coil():
    """Gets the coil-100 data.

    Returns
    -------
        A tuple:
            - imgs1 : The first batch of images
            - imgs2 : The second batch of images
            - c : Number of channels in the images
            - w : The width (and height) of the images
    """
    fp = common.DATA_PROCESSED + 'coil.npz'
    if not os.path.isfile(fp):
        generate_coil_proc()

    cdf = np.load(fp)
    data, labels = cdf['data'], cdf['labels']

    _, w, _, c = data.shape
    data = data.reshape((-1, w * w * c))

    # Just split by rotation level -- 0-175 is set1, 180-355 is set2
    imgs1, imgs2 = [], []
    for index, (_, rotation) in enumerate(labels):
        if int(rotation) < 180:
            imgs1.append(data[index])
        else:
            imgs2.append(data[index])

    # TODO -- i noticed that pixel values for monet ims are being normalized
    # between 0 and 1. not sure if that's necessary here.
    imgs1, imgs2 = np.array(imgs1), np.array(imgs2)
    print(imgs1)
    return imgs1, imgs2, c, w


def generate_coil_proc(cleanup=True):
    """Will convert the zip file downloaded by download_data
    to an npz file.

    Arguments
    ---------
        cleanup (bool) : Delete the raw zip file after we have converted
            it to an npz file.
    """
    zip_fp = common.DATA_RAW + 'coil-100.zip'
    temp_fp = common.DATA_RAW + 'coil-100/'
    if not os.path.isfile(zip_fp):
        raise ValueError("Please run 'download_data coil' to get the appropriate dataset")

    # Unzip images
    zf = zipfile.ZipFile(zip_fp, 'r')
    zf.extractall(common.DATA_RAW)
    zf.close()

    # Convert images to npz file
    ims = list(glob.glob(temp_fp + '*.png'))
    labels = []
    for im in ims:
        labels.append(re.findall('\d+', os.path.basename(im)))
    labels = np.array(labels)
    data = np.stack([scipy.ndimage.imread(f) for f in ims], axis=0)
    np.savez(common.DATA_PROCESSED + 'coil', data=data, labels=labels)

    # Remove temporary files
    if cleanup:
        os.remove(zip_fp)
        shutil.rmtree(temp_fp)


