"""
Structural LSTM model. Performs convolutions on the hidden and cell state
each iteration before sending them into the next cell. The two convolution
outputs are then masked together (Hadamard product) before being sent to the
next cell at the forget and output gates.

This is specifically for 1D inputs (e.g. text).
"""
from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Recurrent
from keras.legacy import interfaces
from keras.utils import conv_utils
import numpy as np

def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x

class StructuralLSTM(Recurrent):
    """
    Implements a Structural LSTM cell, where the hidden state vector is
    over to obtain a structure state vector, which is then fed into the next
    time step. Also has parameters to support the use of a peephole, both with
    keeping and discarding the hidden state vector (the structure state vector
    is always used).

    ###########################################################################
    # This implementation only supports 1D input sequences with a depth of 1! #
    ###########################################################################

    Unlike typical LSTMs, this returns the structure state vector rather than
    the hidden state vector (might change this in the future? idk).

    **Arguments**  
        `units`: Integer >0 specifying the dimensionality of the output space.   
        `filters`: Integer or tuple/list specifying the dimensionality of the
            output space (i.e. the number of output filters in each layer of the
            convolution stack). If an integer, the same number of filters is
            used for all convolution layers.
        `kernel_size`: Integer or tuple/list of integers specifying the
            dimension sizes of the convolution windows. If not a single integer,
            the length of the list-like input should equal to the `conv_layers`
            argument, and each element should be an integer.  
        `return_structure`: If True (default), the cell returns the structure
            state vector as the output, otherwise the hidden state vector is
            returned instead. This doesn't affect the internals of the cell.  
        `peephole`: One of three possible options specifying whether or not the
            cell state vector is fed into each of the gates, and (if so) if the
            hidden state vector is fed with it.  
                `0`: No peeping on the cell state vector (default)  
                `1`: Peep on the cell state vector, including the hidden state  
                `2`: Peep on the cell state vector, but discard the hidden state
        `conv_layers`: Integer >0 specifying the number of convolution layers to
            use. Defaults to 1.  
        `strides`: Integer or tuple/list of integers, specifying the stride
            length of the convolution. If not a single integer, the length of
            the list-like input should equal to the `conv_layers` argument, and
            each element should be an integer. Specifying any `strides` value
            != 1 is incompatible with specifying any `dilation_rate` value != 1.  
        `padding`: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
            `"valid"` means "no padding".  
            `"same"` results in padding the input such that the output has the
                same length as the original input.  
            `"causal"` results in causal (dilated) convolutions, e.g. `output[t]`
                does not depend on `input[t+1:]`. Useful when modeling temporal
                data where the model should not violate the temporal order.  
            See [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499).  
        `dilation_rate`: Integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution. If not a single
            integer, the length of the list-like input should equal to the
            `conv_layers` argument, and each element should be an integer.
            Specifying any `dilation_rate` value != 1 is incompatible with
            specifying any `strides` value != 1.  
        `activation`: Activation function to use for state vectors (see
            [activations](../activations.md)). If None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).  
        `recurrent_activation`: Activation function to use for the recurrent
            steps, specifically the gate activations (see [activations](../activations.md)).  
        `convolution_activation`: Activation function to use for the convolution
            steps (see [activations](../activations.md)).  
        `use_bias`: Boolean specifying whether the layer uses a bias vector (for
            both the recurrent and convolutional portions of the layer).  
        `kernel_initializer`: Initializer for the `kernel` weights matrix, used
            for the linear transformation of the inputs (see
            [initializers](../initializers.md)). Applies to both the recurrent
            and convolutional kernels.  
        `recurrent_initializer`: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).  
        `structure_initializer`: Initializer for the `structure_kernel` weights
            matrix, used for the linear transformation of the structure state.
            (see [initializers](../initializers.md)).  
        `convolution_initializer`:Initializer for the convolution kernel weight
            matrices, used for both the convolution layers and the final fully-
            connected layer. (see [initializers](../initializers.md)).  
        `bias_initializer`: Initializer for the bias vector (see
            [initializers](../initializers.md)). Applies to both the recurrent
            and convolutional biases.  
        `unit_forget_bias`: Boolean. If True, add 1 to the bias of the forget
            gate at initialization. Setting it to true will also force `bias_initializer = "zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).  
        `kernel_regularizer`: Regularizer function applied to the kernel weights
            matrix (see [regularizer](../regularizers.md)).  
        `recurrent_regularizer`: Regularizer function applied to the
            `recurrent_kernel` weights matrix (see [regularizer](../regularizers.md)).  
        `structure_regularizer`: Regularizer function applied to the
            `structure_kernel` weights matrix (see [regularizer](../regularizer.md)).  
        `convolution_regularizer`: Regularizer function applied to the convolution
            kernel weight matrices, used for both the convolution layers and the
            final fully-connected layer (see [regularizer](../regularizer.md)).  
        `bias_regularizer`: Regularizer function applied to the bias vector (see
            [regularizer](../regularizers.md)). Applies to both the recurrent
            and convolutional biases.  
        `activity_regularizer`: Regularizer function applied to the output of
            the layer (its "activation") (see [regularizer](../regularizers.md)).  
        `kernel_constraint`: Constraint function applied to the kernel weights
            matrix (see [constraints](../constraints.md)).  
        `recurrent_constraint`: Constraint function applied to the
            `recurrent_kernel` weights matrix (see [constraints](../constraints.md)).  
        `structure_constraint`: Constraint function applied to the
            `structure_kernel` weights matrix (see [constraints](../constraints.md)).  
        `convolution_constraint`: Constraint function applied to the convolution
            kernel weight matrices, used for both the convolution layers and the
            final fully-connected layer (see [constraints](../constraints.md)).  
        `bias_constraint`: Constraint function applied to the bias vector (see
            [constraints](../constraints.md)). Applies to both the recurrent
            and convolutional biases.  
        `dropout`: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs.  
        `recurrent_dropout`: Float between 0 and 1. Fraction of the units to
            drop for the linear transformation of the recurrent state.  
        `structure_dropout`: Float between 0 and 1. Fraction of the units to
            drop for the linear transformation of the structure state.  
        `peephole_dropout`: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the cell state vector when peeping
            on it. If peephole is not used (`peephole == 0`), this is ignored.  

    NOTE: compute_output_shape() defined in the parent Recurrent class.
    """
    @interfaces.legacy_recurrent_support
    def __init__(self, units, filters, kernel_size,
                 return_structure = True,
                 peephole = 0,
                 conv_layers = 1,
                 strides = 1,
                 padding = 'valid',
                 dilation_rate = 1,
                 activation = 'tanh',
                 recurrent_activation = 'hard_sigmoid',
                 convolution_activation = None,
                 use_bias = True,
                 kernel_initializer = 'glorot_uniform',
                 recurrent_initializer = 'orthogonal',
                 structure_initializer = 'orthogonal',
                 convolution_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 unit_forget_bias = True,
                 kernel_regularizer = None,
                 recurrent_regularizer = None,
                 structure_regularizer = None,
                 convolution_regularizer = None,
                 bias_regularizer = None,
                 activity_regularizer = None,
                 kernel_constraint = None,
                 recurrent_constraint = None,
                 structure_constraint = None,
                 convolution_constraint = None,
                 bias_constraint = None,
                 dropout = 0.,
                 recurrent_dropout = 0.,
                 structure_dropout = 0.,
                 peephole_dropout = 0.,
                 **kwargs):
        # Call extended class initializations
        super(StructuralLSTM, self).__init__(**kwargs)
        conv_layers = max(1, conv_layers)  # Stop people from being sneaky.
        # General shape properties and hyperparameters
        self.units = units
        self.filters = conv_utils.normalize_tuple(
            filters, conv_layers, 'filters'
        )
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, conv_layers, 'kernel_size'
        )
        # What vector to return as the cell output
        self.return_structure = return_structure
        # If peephole is implemented or not
        if peephole not in [0, 1, 2]:
            raise ValueError("peephole can only be 0, 1, or 2.")
        else:
            self.peephole = peephole
        # Convolution hyperparameters
        self.conv_layers = conv_layers
        self.strides = conv_utils.normalize_tuple(
            strides, conv_layers, 'strides'
        )
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, conv_layers, 'dilation_rate'
        )
        # Activation types
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.convolution_activation = activations.get(convolution_activation)
        self.use_bias = use_bias
        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.structure_initializer = initializers.get(structure_initializer)
        self.convolution_initializer = initializers.get(convolution_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias
        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.structure_regularizer = regularizers.get(structure_regularizer)
        self.convolution_regularizer = regularizers.get(convolution_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.structure_constraint = constraints.get(structure_constraint)
        self.convolution_constraint = constraints.get(convolution_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # Dropout
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.structure_dropout = min(1., max(0., structure_dropout))
        self.peephole_dropout = min(1., max(0., peephole_dropout))
        # Create a list of InputSpecs for the structure, hidden, and cell states
        # If not return_structure, order is actually:
        # [hidden state, structure state, cell state]
        self.state_spec = [
            InputSpec(shape = (None, self.units)),  # Structure state
            InputSpec(shape = (None, self.units)),  # Hidden State
            InputSpec(shape = (None, self.units))   # Cell state
        ]

    """
    Create the trainable weights for this layer.
    Individual input_shape is (batch_size, timesteps, dim).
    """
    def build(self, input_shape):
        ###===~~~ State shape specifications ~~~===###
        # Get some dimensions
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        # Create an InputSpec for the input to the layer
        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape = (batch_size, None, self.input_dim))
        ###===~~~ Recurrent gates ~~~===###
        # Hidden, cell, and structure states, respectively
        if self.stateful:
            self.reset_states()
        else:
            # Initial states are three all-zero tensors
            # Structure, hidden, and cell state vectors, respectively
            # If not return_structure, order is actually:
            # [hidden state, structure state, cell state]
            self.states = [None, None, None]
        # Initialize input kernel
        self.kernel = self.add_weight(
            shape = (self.input_dim, self.units*4),
            name = 'kernel',
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            constraint = self.kernel_constraint
        )
        if self.peephole != 2:
            # Initialize recurrent kernel
            self.recurrent_kernel = self.add_weight(
                shape = (self.units, self.units*4),
                name = 'recurrent_kernel',
                initializer = self.recurrent_initializer,
                regularizer = self.recurrent_regularizer,
                constraint = self.recurrent_constraint
            )
        else:
            self.recurrent_kernel = None
        # Initialize structure kernel
        self.structure_kernel = self.add_weight(
            shape = (self.units, self.units*4),
            name = 'structure_kernel',
            initializer = self.structure_initializer,
            regularizer = self.structure_regularizer,
            constraint = self.structure_constraint
        )
        if self.peephole != 0:
            # Initialize peephole kernel
            self.peephole_kernel = self.add_weight(
                shape = (self.units, self.units*4),
                name = 'peephole_kernel',
                initializer = self.recurrent_initializer,
                regularizer = self.recurrent_regularizer,
                constraint = self.recurrent_constraint
            )
        else:
            self.peephole_kernel = None
        # Initialize bias vectors if used
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        # Input bias:
                        self.bias_initializer((self.units,), *args, **kwargs),
                        # Forget bias (ones if unit_forget_bias):
                        initializers.Ones()((self.units,), *args, **kwargs),
                        # Output and cell state biases:
                        self.bias_initializer((self.units*2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape = (self.units*4,),
                name = 'bias',
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                constraint = self.bias_constraint
            )
        else:
            self.bias = None
        # Define kernel subvectors
        self.kernel_i = self.kernel[:,              : self.units]
        self.kernel_f = self.kernel[:, self.units   : self.units*2]
        self.kernel_c = self.kernel[:, self.units*2 : self.units*3]
        self.kernel_o = self.kernel[:, self.units*3 : ]
        if self.recurrent_kernel is not None:
            # Define recurrent kernel subvectors
            self.recurrent_kernel_i = self.recurrent_kernel[:,              : self.units]
            self.recurrent_kernel_f = self.recurrent_kernel[:, self.units   : self.units*2]
            self.recurrent_kernel_c = self.recurrent_kernel[:, self.units*2 : self.units*3]
            self.recurrent_kernel_o = self.recurrent_kernel[:, self.units*3 : ]
        else:
            self.recurrent_kernel_i = None
            self.recurrent_kernel_f = None
            self.recurrent_kernel_c = None
            self.recurrent_kernel_o = None
        # Define structure kernel subvectors
        self.structure_kernel_i = self.structure_kernel[:,              : self.units]
        self.structure_kernel_f = self.structure_kernel[:, self.units   : self.units*2]
        self.structure_kernel_c = self.structure_kernel[:, self.units*2 : self.units*3]
        self.structure_kernel_o = self.structure_kernel[:, self.units*3 : ]
        if self.peephole_kernel is not None:
            # Define peephole kernel subvectors
            self.peephole_kernel_i = self.peephole_kernel[:,              : self.units]
            self.peephole_kernel_f = self.peephole_kernel[:, self.units   : self.units*2]
            self.peephole_kernel_c = self.peephole_kernel[:, self.units*2 : self.units*3]
            self.peephole_kernel_o = self.peephole_kernel[:, self.units*3 : ]
        else:
            self.peephole_kernel_i = None
            self.peephole_kernel_f = None
            self.peephole_kernel_c = None
            self.peephole_kernel_o = None
        # Define bias subvectors if used. Hidden state vector uses no bias.
        if self.use_bias:
            self.bias_i = self.bias[             : self.units]
            self.bias_f = self.bias[self.units   : self.units*2]
            self.bias_c = self.bias[self.units*2 : self.units*3]
            self.bias_o = self.bias[self.units*3 : ]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        ###===~~~ Hidden state vector convolutions ~~~===###
        # Get the dimensions of the input and output to each convolution layer
        # Size of input to the first convolution layer has same size as cell input
        input_dims = [(self.input_dim, 1)]
        output_dims = [self.compute_conv_layer_output(input_dims[0], 0)]
        for i in range(1, self.conv_layers):
            # Get the output dim of the previous convolution layer as the input dim
            input_dims.append(output_dims[i - 1])
            output_dims.append(self.compute_conv_layer_output(input_dims[i], i))
        kernel_shapes = [
            (ks, i[1], o[1]) for ks, i, o in zip(self.kernel_size, input_dims, output_dims)
        ]
        self.kernel_shapes = kernel_shapes
        self.convolution_kernels = [
            self.add_weight(
                shape = self.kernel_shapes[i],
                name = 'convolution_kernel_%d' % i,
                initializer = self.convolution_initializer,
                regularizer = self.convolution_regularizer,
                constraint = self.convolution_constraint
            ) for i in range(self.conv_layers)
        ]
        # Output of convolution stack will be of size:
        #     (batch_size, timesteps, output_dim, nfilters)
        # We want output to be of size:
        #     (batch_size, timesteps, units)
        self.output_reshape_dim = self.filters[-1] * output_dims[-1][0]
        self.output_kernel = self.add_weight(
            shape = (self.filters[-1],) + (output_dims[-1][0], self.units),
            name = 'output_kernel',
            initializer = self.convolution_initializer,
            regularizer = self.convolution_regularizer,
            constraint = self.convolution_constraint
        )
        if self.use_bias:
            self.convolution_bias = [
                self.add_weight(
                    shape = (self.filters[i],),
                    name = 'convolution_bias_%d' % i,
                    initializer = self.bias_initializer,
                    regularizer = self.bias_regularizer,
                    constraint = self.bias_constraint
                ) for i in range(self.conv_layers)
            ]
            self.bias_s = self.add_weight(
                shape = (self.units,),
                name = 'structure_bias',
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                constraint = self.bias_constraint
            )
        else:
            self.hidden_conv_bias = None
            self.bias_s = None
        self.built = True

    def preprocess_input(self, inputs, training=None):
        if self.implementation == 0:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_i = _time_distributed_dense(inputs, self.kernel_i, self.bias_i,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training = training)
            x_f = _time_distributed_dense(inputs, self.kernel_f, self.bias_f,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training = training)
            x_c = _time_distributed_dense(inputs, self.kernel_c, self.bias_c,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training = training)
            x_o = _time_distributed_dense(inputs, self.kernel_o, self.bias_o,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training = training);
            return K.concatenate([x_i, x_f, x_c, x_o], axis = 2)
        else:
            return inputs

    """
    Calculates the output shape of convolutional layer i.
    """
    def compute_conv_layer_output(self, input_dim, i):
        new_dim = conv_utils.conv_output_length(
            input_dim[0], self.kernel_size[i], padding = self.padding,
            stride = self.strides[i], dilation = self.dilation_rate[i]
        )
        return (new_dim, self.filters[i])

    def reset_states(self, states = None):
        if not self.stateful:
            raise AttributeError("Layer must be stateful.")
        batch_size = self.input_spec[0].shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
        # Initialize states if None
        if self.states[0] is None:
            self.states = [
                K.zeros((batch_size, self.units)) for _ in self.states
            ]
        elif states is None:
            for state in self.states:
                K.set_value(state, np.zeros((batch_size, self.units)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if value.shape != (batch_size, self.units):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, self.units)) +
                                     ', found shape=' + str(value.shape))
                K.set_value(state, value)

    def get_constants(self, inputs, training = None):
        constants = []
        if self.implementation != 0 and 0 < self.dropout < 1:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():  # Define dropout mask function
                return K.dropout(ones, self.dropout)

            dp_mask = [
                K.in_train_phase(
                    dropped_inputs, ones, training = training
                ) for _ in range(4)
            ]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if self.recurrent_kernel is not None:
            if 0 < self.recurrent_dropout < 1:
                ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, self.units))

                def dropped_inputs():  # Define dropout mask function
                    return K.dropout(ones, self.recurrent_dropout)

                rec_dp_mask = [
                    K.in_train_phase(
                        dropped_inputs, ones, training = training
                    ) for _ in range(4)
                ]
                constants.append(rec_dp_mask)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        else:
            constants.append(None)

        if 0 < self.structure_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():  # Define dropout mask function
                return K.dropout(ones, self.structure_dropout)

            str_dp_mask = [
                K.in_train_phase(
                    dropped_inputs, ones, training = training
                ) for _ in range(4)
            ]
            constants.append(str_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if self.peephole_kernel is not None:
            if 0 < self.peephole_dropout < 1:
                ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, self.units))

                def dropped_inputs():  # Define dropout mask function
                    return K.dropout(ones, self.peephole_dropout)

                peep_dp_mask = [
                    K.in_train_phase(
                        dropped_inputs, ones, training = training
                    ) for _ in range(4)
                ]
                constants.append(peep_dp_mask)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        else:
            constants.append(None)

        return constants

    def step(self, inputs, states):
        #=~ Get previous iteration's state vectors and dropout masks ~=#
        if self.return_structure:
            s_tm1 = states[0]
            h_tm1 = states[1]
        else:
            h_tm1 = states[0]
            s_tm1 = states[1]
        c_tm1 = states[2]
        dp_mask = states[3]
        rec_dp_mask = states[4]
        str_dp_mask = states[5]
        peep_dp_mask = states[6]

        #=~ Calculate the hidden and cell state vectors for this iteration ~=#
        # Calculate the outputs of the forget, input, and output gates, as well
        # as the cell state vector, based on the preferred implementation type.
        if self.implementation == 2:
            z = K.dot(inputs * dp_mask[0], self.kernel)
            if self.recurrent_kernel is not None:
                z += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
            z += K.dot(s_tm1 * str_dp_mask[0], self.structure_kernel)
            if self.peephole_kernel is not None:
                z += K.dot(c_tm1 * peep_dp_mask[0], self.peephole_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z_i = z[:,              : self.units]
            z_f = z[:, self.units   : self.units*2]
            z_c = z[:, self.units*2 : self.units*3]
            z_o = z[:, self.units*3 : ]

            i = self.recurrent_activation(z_i)
            f = self.recurrent_activation(z_f)
            c = f * c_tm1 + i * self.activation(z_c)
            o = self.recurrent_activation(z_o)
        else:
            if self.implementation == 0:
                x_i = inputs[:,              : self.units]
                x_f = inputs[:, self.units   : self.units*2]
                x_c = inputs[:, self.units*2 : self.units*3]
                x_o = inputs[:, self.units*3 : ]
            elif self.implementation == 1:
                x_i = K.dot(inputs * dp_mask[0], self.kernel_i) + self.bias_i
                x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
                x_c = K.dot(inputs * dp_mask[2], self.kernel_c) + self.bias_c
                x_o = K.dot(inputs * dp_mask[3], self.kernel_o) + self.bias_o

            else:
                raise ValueError("Unknown `implementation` mode %d." % self.implementation)

            _sub_i = K.dot(s_tm1 * str_dp_mask[0], self.structure_kernel_i)
            if self.recurrent_kernel is not None:
                _sub_i += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel_i)
            if self.peephole_kernel is not None:
                _sub_i += K.dot(c_tm1 * peep_dp_mask[0], self.peephole_kernel_i)
            i = self.recurrent_activation(x_i + _sub_i)
            
            _sub_f = K.dot(s_tm1 * str_dp_mask[1], self.structure_kernel_f)
            if self.recurrent_kernel is not None:
                _sub_f += K.dot(h_tm1 * rec_dp_mask[1], self.recurrent_kernel_f)
            if self.peephole_kernel is not None:
                _sub_f += K.dot(c_tm1 * peep_dp_mask[1], self.peephole_kernel_f)
            f = self.recurrent_activation(x_f + _sub_f)

            _sub_c = K.dot(s_tm1 * str_dp_mask[2], self.structure_kernel_c)
            if self.recurrent_kernel is not None:
                _sub_c += K.dot(h_tm1 * rec_dp_mask[2], self.recurrent_kernel_c)
            if self.peephole_kernel is not None:
                _sub_c += K.dot(c_tm1 * peep_dp_mask[2], self.peephole_kernel_c)
            c = f * c_tm1 + i * self.activation(x_c + _sub_c)

            _sub_o = K.dot(s_tm1 * str_dp_mask[3], self.structure_kernel_o)
            if self.recurrent_kernel is not None:
                _sub_o += K.dot(h_tm1 * rec_dp_mask[3], self.recurrent_kernel_o)
            if self.peephole_kernel is not None:
                _sub_o += K.dot(c_tm1 * peep_dp_mask[3], self.peephole_kernel_c)
            o = self.recurrent_activation(x_o + _sub_o)
        # Calculate new hidden state vector
        h = o * self.activation(c)
        #=~ Calculate the structure state vector for this iteration ~=#
        # Prep the convolution of the hidden state vector by giving it depth
        deep_h = K.expand_dims(h)
        # Perform the actual convolutions for each convolution layer
        for nl in range(self.conv_layers):
            # Hidden state vector convolutions
            deep_h = K.conv1d(
                deep_h,
                self.convolution_kernels[nl],
                strides = self.strides[nl],
                padding = self.padding,
                data_format = 'channels_last',
                dilation_rate = self.dilation_rate[nl]
            )
            if self.use_bias:
                deep_h =K.bias_add(
                    deep_h,
                    self.convolution_bias[nl],
                    data_format = 'channels_last'
                )
            if self.convolution_activation is not None:
                deep_h = self.convolution_activation(deep_h)
        # Calculate the structure state vector using the convolution output
        deep_h_shape = K.int_shape(deep_h)
        s = self.activation(
            #TODO: Can't do tensor dot products contracting over two dimensions
            # like this. Need to fix.
            K.dot(
                K.reshape(deep_h, deep_h_shape[:-2] + (self.output_reshape_dim,)),
                K.reshape(self.output_kernel, (self.output_reshape_dim, self.units))
            ) + self.bias_s
        )
        # Set learning phase flag if dropout has been used
        dp = self.dropout + self.structure_dropout
        if self.recurrent_kernel is not None:
            dp += self.recurrent_dropout
        if self.peephole_kernel is not None:
            dp += self.peephole_dropout
        if 0 < dp:
            s._uses_learning_phase = True
        if self.return_structure:
            return s, [s, h, c]
        else:
            return h, [h, s, c]

    def get_config(self):
        config = {
            'units': self.units,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'return_structure': self.return_structure,
            'peephole': self.peephole,
            'conv_layers': self.conv_layers,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'convolution_activation': activations.serialize(self.convolution_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'structure_initializer': initializers.serialize(self.structure_initializer),
            'convolution_initializer': initializers.serialize(self.convolution_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'unit_forget_bias': self.unit_forget_bias,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'structure_regularizer': regularizers.serialize(self.structure_regularizer),
            'convolution_regularizer': regularizers.serialize(self.convolution_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
            'structure_constraint': constraints.serialize(self.structure_constraint),
            'convolution_constraint': constraints.serialize(self.convolution_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'structure_dropout': self.structure_dropout,
            'peephole_dropout': self.peephole_dropout
        }
        base_config = super(StructuralLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

