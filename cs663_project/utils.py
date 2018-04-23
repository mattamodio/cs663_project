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
    'model_out', 'save_iter',
    'early_stopping_enabled', 'early_stopping_patience'
])


class EarlyStopping(object):
    """
    Provides early stopping functionality. Keeps track of model accuracy,
    and if it doesn't improve over time restores last best performing
    parameters.

    credit: @navoshta
    """

    def __init__(self, saver, session, patience = 100, minimize = True):
        """
        Initialises a `EarlyStopping` isntance.

        Parameters
        ----------
        saver     :
                    TensorFlow Saver object to be used for saving and restoring model.
        session   :
                    TensorFlow Session object containing graph where model is restored.
        patience  :
                    Early stopping patience. This is the number of iterations we wait for
                    accuracy to start improving again before stopping and restoring
                    previous best performing parameters.

        Returns
        -------
        New instance.
        """
        self.minimize = minimize
        self.patience = patience
        self.saver = saver
        self.session = session
        self.best_monitored_dloss = np.inf if minimize else 0.
        self.best_monitored_gloss = np.inf if minimize else 0.
        self.best_monitored_iteration = 0
        self.restore_path = None

    def __call__(self, d_loss, g_loss, iteration):
        """
        Checks if we need to stop and restores the last well performing values if we do.

        Parameters
        ----------
            d_loss (float) : Last iteration monitored discriminator loss.
            g_loss (float) : Last iteration monitored generator loss.
            iteration (int) : Last iteration number.

        Returns
        -------
            `True` if we waited enough and it's time to stop and we restored the
            best performing weights, or `False` otherwise.
        """
        if ((self.minimize and
             dloss < self.best_monitored_dloss and
             gloss < self.best_monitored_gloss) or
            (not self.minimize and
             dloss > self.best_monitored_dloss and
             gloss > self.best_monitored_gloss)
           ):
            self.best_monitored_dloss = dloss
            self.best_monitored_gloss = gloss
            self.best_monitored_iteration = iteration
            self.restore_path = self.saver.save(
                self.session, os.getcwd() + "/early_stopping_checkpoint")
        elif self.best_monitored_iteration + self.patience < iteration:
            if self.restore_path != None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print("ERROR: Failed to restore session")
            return True

        return False


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

