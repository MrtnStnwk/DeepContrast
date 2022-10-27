from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Model
from tensorflow.python.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import load_model
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np

import metrics

def get_channel_axis(data_format):
    """Get the channel axis given the data format
    :param str data_format: as named. [channels_last, channel_first]
    :return int channel_axis
    """
    assert data_format in ['channels_first', 'channels_last'], \
    	'Invalid data format %s' % data_format
    if data_format == 'channels_first':
    	channel_axis = 1
    else:	
    	channel_axis = -1
    return channel_axis



def _split_ytrue_mask(y_true, n_channels, n_masks=1):
    """Split the mask concatenated with y_true
    :param keras.tensor y_true: if channels_first, ytrue has shape [batch_size,
     n_channels, y, x]. mask is concatenated as the n_channels+1, shape:
     [[batch_size, n_channels+1, y, x].
    :param int n_channels: number of channels in y_true
    :return:
     keras.tensor ytrue_split - ytrue with the mask removed
     keras.tensor mask_image - bool mask
    """

    try:
        split_axis = get_channel_axis(K.image_data_format())
        print('splitted',y_true)
        y_true_split, mask_image = tf.split(y_true, [n_channels, n_masks],
                                            axis=split_axis)
        return y_true_split, mask_image
    except Exception as e:
        print('cannot separate mask and y_true' + str(e))


def mae_loss(y_true, y_pred, mean_loss=True):
    """Mean absolute error
    Keras losses by default calculate metrics along axis=-1, which works with
    image_format='channels_last'. The arrays do not seem to batch flattened,
    change axis if using 'channels_first
    """
    if not mean_loss:
        return K.abs(y_pred - y_true)

    channel_axis = get_channel_axis(K.image_data_format())
    return K.mean(K.abs(y_pred - y_true), axis=channel_axis)

