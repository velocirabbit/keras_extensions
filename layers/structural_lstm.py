"""
Structural LSTM model. Similar to the basic LSTM model, but also takes the cell
state vector and uses it as the input to a cell-dependent Markov network, the
output of which is taken as the structure state vector and fed recursively into
the next time step of the cell where it's weighted and added to the next input.

Also allows for the use of cell state vector peepholes (both with the hidden
state vector included, and with it excluded).
"""
from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Dense, Recurrent
from keras.legacy import interfaces
import numpy as np

class StructuralLSTM(Recurrent):
    """
    Implements a Structural LSTM cell, where the cell state vector is used as
    the input to a cell-dependent Markov network, the output of which is taken
    as the structure state vector.

    **Arguments**  
        `units`: Integer >0 specifying the dimensionality of the output space.  
        `return_structure`: If True (default), the cell returns the structure
            state vector as the output, otherwise the hidden state vector is
            returned instead. This doesn't affect the internals of the cell.  
        `peephole`: One of four possible options specifying whether or not the
            cell state vector is fed into each of the gates, and (if so) if the
            hidden state vector is fed with it, or if both are ignored.  
                `0`: No peeping on the cell state vector (default)  
                `1`: Peep on the cell state vector, including the hidden state  
                `2`: Peep on the cell state vector, but discard the hidden state  
                `3`: Ignore both the hidden and cell state vectors  
        `num_samples`: Number of Gibbs samples to take each batch when
            performing signal/structure output reconstruction.  
        `activation`: Activation function to use for state vectors, including
            feeding the output vector back into itself during the Gibbs sampling
            reconstruction step in the cell-dependent Markov network. If None,
            no activation is applied (ie. "linear" activation: `a(x) = x`).  
        `recurrent_activation`: Activation function to use for the recurrent
            steps, specifically the gate activations.  
        `signal_activation`: Activation function to use on the weighted input
            signals to the cell-dependent Markov network.  
        `use_bias`: Boolean specifying whether the layer uses bias vectors.  
        `kernel_initializer`: Initializer for the `kernel` weights matrix, used
            for the linear transformation of the inputs.    
        `recurrent_initializer`: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.  
        `structure_initializer`: Initializer for the `structure_kernel` weights
            matrix, used for the linear transformation of the structure state.  
        `signal_initializer`: Initializer for the `signal_kernel` weights matrix 
            of the cell-dependent Markov network.  
        `feedback_initializer`: Initializer for the `feedback_kernel` weights
            matrix used to find intra-neuron correlations during the Gibbs
            sampling reconstruction of the output vector in the cell-dependent
            Markov network.  
        `bias_initializer`: Initializer for the bias vectors.  
        `unit_forget_bias`: Boolean. If True, add 1 to the bias of the forget
            gate at initialization. Setting it to true will also force
            `bias_initializer = "zeros"`.  
        `kernel_regularizer`: Regularizer function applied to the kernel weights
            matrix.  
        `recurrent_regularizer`: Regularizer function applied to the
            `recurrent_kernel` weights matrix.  
        `structure_regularizer`: Regularizer function applied to the
            `structure_kernel` weights matrix.  
        `signal_regularizer`: Regularizer function applied to the `signal_kernel`
            weights matrix of the cell-dependent Markov network.  
        `feedback_regularizer`: Regularizer function applied to the
            `feedback_kernel` weights matrix in the cell-dependent Markov
            network.
        `bias_regularizer`: Regularizer function applied to the bias vector.  
        `activity_regularizer`: Regularizer function applied to the output of
            the layer (its "activation").  
        `kernel_constraint`: Constraint function applied to the kernel weights
            matrix.  
        `recurrent_constraint`: Constraint function applied to the
            `recurrent_kernel` weights matrix.  
        `structure_constraint`: Constraint function applied to the
            `structure_kernel` weights matrix.  
        `signal_constraint`: Constraint function applied to the `signal_kernel` weights
            matrix of the state-dependent Markov network.  
        `feedback_constraint`: Constraint function applied to the
            `constraint_kernel` weights matrix of the state-dependent Markov
            network.  
        `bias_constraint`: Constraint function applied to the bias vector.  
        `dropout`: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs.  
        `recurrent_dropout`: Float between 0 and 1. Fraction of the units to
            drop for the linear transformation of the recurrent state.  
        `structure_dropout`: Float between 0 and 1. Fraction of the units to
            drop for the linear transformation of the structure state.  
        `peephole_dropout`: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the cell state vector when peeping
            on it. If peephole is not used (`peephole == 0`), this is ignored.  
        `signal_dropout`: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the input signal to the cell-
            dependent Markov network.  
        `feedback_dropout`: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the previous structure state vector
            used when performing Gibbs sampling reconstruction of the next
            structure state output vector.  
    """
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 return_structure = True,
                 peephole = 0,  
                 num_samples = 1,
                 # Activations  
                 activation = 'tanh',
                 recurrent_activation = 'hard_sigmoid',
                 signal_activation = 'selu',
                 use_bias = True,  
                 # Initializers  
                 kernel_initializer = 'glorot_uniform',
                 recurrent_initializer = 'orthogonal',
                 structure_initializer = 'orthogonal',
                 signal_initializer = 'glorot_uniform',
                 feedback_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 unit_forget_bias = True,  
                 # Regularizers  
                 kernel_regularizer = None,
                 recurrent_regularizer = None,
                 structure_regularizer = None,
                 signal_regularizer = None,
                 feedback_regularizer = None,
                 bias_regularizer = None,
                 activity_regularizer = None,  
                 # Contraints  
                 kernel_constraint = None,
                 recurrent_constraint = None,
                 structure_constraint = None,
                 signal_constraint = None,
                 feedback_constraint = None,
                 bias_constraint = None,  
                 # Dropout rates  
                 dropout = 0.,
                 recurrent_dropout = 0.,
                 structure_dropout = 0.,
                 peephole_dropout = 0.,
                 signal_dropout = 0.,
                 feedback_dropout = 0.,
                 # Other keyword arguments  
                 **kwargs):
        # Call extended class initializations
        super(StructuralLSTM, self).__init__(**kwargs)
        # Length of state vectors
        self.units = units
        # What vector to return as the cell output
        self.return_structure = return_structure
        # If peephole is implemented or not
        if peephole not in [0, 1, 2, 3]:
            raise ValueError("peephole can only be 0, 1, or 2.")
        else:
            self.peephole = peephole
        self.num_samples = num_samples
        # Activation types
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.signal_activation = activations.get(signal_activation)
        self.use_bias = use_bias
        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.structure_initializer = initializers.get(structure_initializer)
        self.signal_initializer = initializers.get(signal_initializer)
        self.feedback_initializer = initializers.get(feedback_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias
        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.structure_regularizer = regularizers.get(structure_regularizer)
        self.signal_regularizer = regularizers.get(signal_regularizer)
        self.feedback_regularizer = regularizers.get(feedback_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.structure_constraint = constraints.get(structure_constraint)
        self.signal_constraint = constraints.get(signal_constraint)
        self.feedback_constraint = constraints.get(feedback_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # Dropout
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.structure_dropout = min(1., max(0., structure_dropout))
        self.peephole_dropout = min(1., max(0., peephole_dropout))
        self.signal_dropout = min(1., max(0., signal_dropout))
        self.feedback_dropout = min(1., max(0., feedback_dropout))
        # Create a list of InputSpecs for the structure, hidden, and cell states
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
            self.states = [None, None, None]
        # Initialize input kernel
        self.kernel = self.add_weight(
            shape = (self.input_dim, self.units*4),
            name = 'kernel',
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            constraint = self.kernel_constraint
        )
        if self.peephole < 2:
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
        if 0 < self.peephole < 3:
            # Initialize cell state peephole kernel
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
        ###===~~~ Cell-dependent Markov network sub-unit ~~~===###
        # Define the filer matrix applied over the input signal
        self.signal_kernel = self.add_weight(
            shape = (self.units, self.units),
            name = 'signal_kernel',
            initializer = self.signal_initializer,
            regularizer = self.signal_regularizer,
            constraint = self.signal_constraint,
            trainable = False   # We won't train these weights during backprop
        )
        # Define RBM biases together. Need one for the visible states, and one
        # for the hidden states.
        self.signal_bias = self.add_weight(
            shape = (self.units*2,),
            name = 'signal_bias',
            initializer = self.bias_initializer,
            regularizer = self.bias_regularizer,
            contraint = self.bias_constraint,
            trainable = False   # We won't train these weights during backprop
        )
        # Define the RBM bias subvectors
        self.signal_bias_g = self.signal_bias[           : self.units]
        self.signal_bias_s = self.signal_bias[self.units :           ]
        # Define the intra-neural correlation weights matrix
        self.feedback_kernel = self.add_weight(
            shape = (self.units, self.units),
            name = 'feedback_kernel',
            initializer = self.feedback_initializer,
            regularizer = self.feedback_regularizer,
            constraint = self.feedback_constraint,
            trainable = False   # We won't train these weights during backprop
        )
        # Mark the model as built
        self.built = True

    """
    Gets the various dropout masks, returning a list of them.
    Input shape: `(samples, time, input_dim)`
    """
    def get_constants(self, inputs, training = None):
        constants = []
        #=~ Input vector dropout mask ~=#
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
        #=~ Hidden state recurrence dropout mask ~=#
        if self.recurrent_kernel is not None:
            if 0 < self.recurrent_dropout < 1:
                ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, self.units))  # (samples, units)

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
        #=~ Structure state recurrence dropout mask ~=#
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
        #=~ Cell state recurrence peephole dropout mask ~=#
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
        #=~ Input signal filter dropout mask ~=#
        if 0 < self.signal_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():  # Define dropout mask function
                return K.dropout(ones, self.signal_dropout)

            sgn_dp_mask = K.in_train_phase(
                dropped_inputs, ones, training = training
            )
            constants.append(sgn_dp_mask)
        else:
            constants.append(K.cast_to_floatx(1.))
        #=~ Structure state feedback dropout mask ~=#
        if 0 < self.feedback_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():  # Define dropout mask function
                return K.dropout(ones, self.feedback_dropout)

            fb_dp_mask = K.in_train_phase(
                dropped_inputs, ones, training = training
            )
            constants.append(fb_dp_mask)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def step(self, inputs, states):
        #=~ Get previous iteration's state vectors and dropout masks ~=#
        # State vectors
        s_tm1 = states[0]           # Structure state vector
        h_tm1 = states[1]           # Hidden state vector
        c_tm1 = states[2]           # Cell state vector
        # Kernel dropout masks
        dp_mask = states[3]         # Input kernel dropout mask
        rec_dp_mask = states[4]     # Hidden state recurrence dropout mask
        str_dp_mask = states[5]     # Structure state recurrence dropout mask
        peep_dp_mask = states[6]    # Cell state peephole dropout mask
        sgn_dp_mask = states[7]     # Signal filter dropout mask
        fb_dp_mask = states[8]      # Structure state feedback dropout mask

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
        # Intermediate vector calculation
        signal = self.activation(c)
        # Calculate new hidden state vector
        h = o * signal
        #=~ Cell-dependent Markov network sub-unit ~=#
        # Run Gibbs sampling for num_samples steps
        sgn_i = signal
        for i in range(self.num_samples):
            # Calculate the generating signal, g
            g = self.signal_activation(
                K.dot(sgn_i * sgn_dp_mask, self.signal_kernel) + self.signal_bias_g
            )
            # Sample the (hidden) structure from the (visible) generated signal
            s_rate = K.sigmoid(
                K.dot()  #TODO: Finish this! Need to rederive probabilities
            )
        # Perform one more inference of the structure state vector, this time
        # keeping it as a tensor of probabilities rather than binary sampling
        s = None  # Structure state vector
        # Set learning phase flag if any dropout has been used
        dp = self.dropout + self.structure_dropout + self.signal_dropout + self.feedback_dropout
        if self.recurrent_kernel is not None:
            dp += self.recurrent_dropout
        if self.peephole_kernel is not None:
            dp += self.peephole_dropout
        if self.return_structure:
            if dp > 0:
                s._uses_learning_phase = True
            return s, [s, h, c]
        else:
            if dp > 0:
                h._uses_learning_phase = True
            return h, [s, h, c]

    def get_config(self):
        config = {
            'units': self.units,
            'return_structure': self.return_structure,
            'peephole': self.peephole,
            'num_samples': self.num_samples,
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'signal_activation': activations.serialize(self.signal_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'structure_initializer': initializers.serialize(self.structure_initializer),
            'signal_initializer': initializers.serialize(self.signal_initializer),
            'feedback_initializer': initializers.serialize(self.feedback_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'unit_forget_bias': self.unit_forget_bias,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'structure_regularizer': regularizers.serialize(self.structure_regularizer),
            'signal_regularizer': regularizers.serialize(self.signal_regularizer),
            'feedback_regularizer': regularizers.serialize(self.feedback_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
            'structure_constraint': constraints.serialize(self.structure_constraint),
            'signal_contraint': constraints.get(self.signal_constraint),
            'feedback_constraint': constraints.get(self.feedback_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'structure_dropout': self.structure_dropout,
            'peephole_dropout': self.peephole_dropout,
            'signal_dropout': self.signal_dropout,
            'feedback_dropout': self.feedback_dropout
        }
        base_config = super(StructuralLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

