from collections import namedtuple
import datetime
import math
import tensorflow as tf
import numpy as np


# Parameters for the neural network
DiscoConfig = namedtuple('DiscoConfig', [
    # Data parameters
    'num_classes', 'image_size', 'channels', 'data',
    # Training parameters
    'batch_size', 'num_iters',
    # Optimizations
    'learning_rate', 'attn', 'decay',
    'wasserstein',
    # Network parameters
    # Meta parameters
    'model_out', 'save_iter'
])


def now():
    return datetime.datetime.now().strftime('%m-%d %H:%M:%S')

def asinh(x, scale=5.):
    f = np.vectorize(lambda y: math.asinh(y / scale))

    return f(x)

def sinh(x, scale=5.):
    return scale * np.sinh(x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape((-1, 1))

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def nameop(op, name):
    op = tf.identity(op, name=name)
    return op

def tbn(name):
    return tf.get_default_graph().get_tensor_by_name(name)

def obn(name):
    return tf.get_default_graph().get_operation_by_name(name)

def get_all_node_names():
    return [n.name for n in tf.get_default_graph().as_graph_def().node]

