
"""Custom metrics"""
from tensorflow.python.keras import backend as K
import tensorflow as tf

def flip_dimensions(func):
    """
    Decorator to convert channels first tensor to channels last.
    :param func: Function to be decorated
    """
    def wrap_function(y_true, y_pred):
        """
        Shifts dimensions to channels last if applicable, before
        calling func.
        :param tensor y_true: Gound truth data
        :param tensor y_pred: Predicted data
        :return: function called with channels last
        """
        if K.image_data_format() == 'channels_first':
            if K.ndim(y_true) > 4:
                y_true = tf.transpose(y_true, [0, 2, 3, 4, 1])
                y_pred = tf.transpose(y_pred, [0, 2, 3, 4, 1])
            else:
                y_true = tf.transpose(y_true, [0, 2, 3, 1])
                y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
        return func(y_true, y_pred)
    return wrap_function

