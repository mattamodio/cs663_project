import abc

import tensorflow as tf


class SimilarityLoss(abc.ABC):
    """Abstract base class for similarity loss. An ABC may be overkill
    right now but could come in handy if we define more elaborate loss
    functions.
    """

    @abc.abstractmethod
    def __call__(self, X_real, X_fake):
        """Compute the similarity loss between two images.

        Arguments
        ---------
            X_real (np.array) : An image from domain A.
            X_fake (np.array) : The mapped image from domain B.

        Returns
        -------
            Float representing the similarity loss.
        """
        pass


class TrivialSimilarityLoss(SimilarityLoss):

    def __call__(self, X_real, X_fake):
        """Example to show usage of the base class. This adds nothing to the
        loss.
        """
        return tf.reduce_mean(tf.zeros_like(X_real))
