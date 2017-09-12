"""
Attention layer for seq2seq autoencoding models. This implements the global
attention model as described in "Effective Approaches to Attention-based Neural
Machine Translation" (Luong, Pham, Manning; 2015).
"""
###===~~~ Imports ~~~===###
from keras import constraints, initializers, regularizers
import keras.backend as K
from keras.engine import Layer

import numpy as np
import tensorflow as tf

class Attention(Layer):
        def __init__(self,
                     kernel_initializer = 'glorot_uniform',
                     kernel_regularizer = None,
                     kernel_constraint = None,
                     dropout = 0.,
                     **kwargs):
            super(Attention, self).__init__(**kwargs)
            self.supports_masking = True
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.kernel_constraint = constraints.get(kernel_constraint)
            self.dropout = dropout
            
        def build(self, input_shape):
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            dim = input_shape[2]
            self.kernel = self.add_weight(
                name = 'kernel',
                shape = (dim, dim),
                initializer = self.kernel_initializer,
                regularizer = self.kernel_regularizer,
                constraint = self.kernel_constraint
            )
            super(Attention, self).build(input_shape)
            
        #TODO: Implement in-layer dropout
        
        '''
        For each timestep, calculate the alignment weights using the current
        timestep plus each timestep before it, then use the weights to get
        the context vector for that timestep.
        '''
        def call(self, x, mask = None):
            if mask is not None:
                # mask (batch, time)
                mask = K.cast(mask, K.floatx())
                # mask (batch, x_dim, time)
                mask = K.repeat(mask, x.shape[-1])
                # mask (batch, time, x_dim)
                mask = tf.transpose(mask, [0, 2, 1])
                x = x * mask
            timesteps = x.shape[1]
            #=~ Calculate alignment weights for each batch ~=#
            # Compute pairwise scores
            align_wts = K.batch_dot(K.dot(x, self.kernel), x, axes = 2)
            # Reduce softmax errors by reducing exponent magnitudes
            align_wts = tf.matrix_band_part(align_wts, 0, -1)
            if mask is not None:  
                align_wts -= K.sum(align_wts, axis = 1) / K.sum(mask, axis = 1)
            else:
                align_wts -= K.mean(align_wts, axis = 1)
            # Compute softmax
            align_wts = tf.matrix_band_part(K.exp(tf.matrix_band_part(align_wts, 0, -1)), 0, -1)
            align_wts /= K.sum(align_wts, axis = 1)
            #=~ Calculate the context tensors for each batch ~=#
            # x has shape (b, t, d); align_wts has shape (b, t, t)
            context = K.batch_dot(x, align_wts, axes = [1, 1])  # (b, d, t)
            context = K.permute_dimensions(context, (0, 2, 1))  # (b, t, d)
            return context