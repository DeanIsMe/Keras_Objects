# -*- coding: utf-8 -*-
"""
Clockwork RNN implementation on Keras (Tensorflow)
Based off the paper by
Jan Koutník, Klaus Greff, Faustino Gomez, Jürgen Schmidhuber
Retrieved from https://arxiv.org/abs/1402.3511
Implementation taking inspiration from chaxor's comment on this thread:
https://github.com/keras-team/keras/issues/2669

Tested using Keras 2.1.3, Tensorflow 1.5
Example usage:
model.add(CWRNN(units=neurons, clock_periods=[1,2,4,8,16], split_method=False, batch_input_shape=(samples, 1, inFeatures), stateful=True))
Made by Dean Reading, 03/2018 
"""
import warnings
import numpy as np
import keras.backend as K
from keras.layers.recurrent import RNN, _generate_dropout_mask, _generate_dropout_ones
from keras.engine import Layer
from keras import activations, regularizers, constraints, initializers
from keras.legacy import interfaces
import tensorflow as tf

class CWRNNCell(Layer):
    """Cell class for CWRNN
    Based off SimpleRNNCell from Keras
    Made by Dean Reading, 2018
    
    # Arguments
        units: Positive integer, dimensionality of the output space.
        clock_periods: A list of the clock periods for each 'module'. 
            In the paper, it's [1,2,4,8,16].
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        split_method:
            I implemented the CWRNN via 2 different methods, which should
            (theoretically) be equivalent. The split method updates each module
            1-by-1, and the standard method updates all weights at once. The 
            split method turned out to be slower, so I don't use it by default.
            But, I've included it for evaluation / discussion / validation.
    """

    def __init__(self, units, clock_periods,
                 activation='tanh',
                 recurrent_activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 split_method = False,
                 **kwargs):
        super(CWRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = [self.units, 1]
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        
        # Fields added for CWRNN
        self.clock_numPeriods = len(clock_periods)
        assert units % self.clock_numPeriods == 0, ("CWRNN requires the units to be " +
                                                "a multiple of the number of periods; " +
                                                "units %% len(clock_periods) must be 0.")
        self.connection_mask = None
        self.clock_periods = np.asarray(sorted(clock_periods))
        self.split_method = split_method
        

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 2),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        # Added for CWRNN
        # Could be made more efficient.
        # Not all of recurrent_kernel_c are used. But, to make it more efficient
        # I believe that I'd need to make the last dimension equal to unitsPerMod,
        # instead of units. This means that I wouldn't be able to update as
        # much with the same calculation
        self.recurrent_kernel_c = self.recurrent_kernel[:, :self.units] # State to state
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units:self.units*2] # State to output
        self.kernel_c = self.kernel[:, :self.units] # Input to state
        
        if self.split_method:
            # Split weights into modules
            unitsPerMod = self.units // self.clock_numPeriods
            self.rec_kernel_c_mod = [0] * self.clock_numPeriods
            self.rec_kernel_o_mod = [0] * self.clock_numPeriods
            self.kernel_c_mod = [0] * self.clock_numPeriods
            self.bias_mod = [0] * self.clock_numPeriods
            for i in range(self.clock_numPeriods):
                s = i*unitsPerMod
                e = (i+1)*unitsPerMod
                self.rec_kernel_c_mod[i] = self.recurrent_kernel_c[s:, s:e]
                self.rec_kernel_o_mod[i] = self.recurrent_kernel_o[:, s:e]
                self.kernel_c_mod[i] = self.kernel_c[:, s:e]
                if self.use_bias:
                    self.bias_mod[i] = self.bias[s:e]
        
        # Create Clockwork RNN constants
        # Each of the neurons (units) has an assigned clock rate / period
        # The neuron is activated every <period> timesteps.
        # Each neuron is only connected to neurons with the same or slower
        # clock rates. mask enforces this rule by controlling connections
        n = self.units // len(self.clock_periods) # n = module length
        cw_mask = np.zeros((self.units, self.units), K.floatx())
        cw_periods = np.zeros((self.units,), np.int32)
        for i, per in enumerate(self.clock_periods):
            # rec_kernel_c[Col x, Row y]  is the influence of state y on state x
            # Higher-module states are not influenced by lower-module states
            # So, rec_kernel has zeros in top-right
            cw_mask[i*n:(i+1)*n, :(i+1)*n] = 1
            cw_periods[i*n:(i+1)*n] = per
        self.cw_mask = K.variable(cw_mask, name='clockwork_mask')
        self.cw_periods = K.variable(cw_periods, dtype='float32', name='clockwork_period')
        
        self.built = True

    def call(self, inputs, states, training=None):
        samples, inFeatures = states[0].shape
        h_tm1 = states[0] # previous state
        time_step = states[1]
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            inputs *= dp_mask
            
        if rec_dp_mask is not None:
            h_tm1 *= rec_dp_mask

        if self.split_method:
            # Update State, module-by-module
            h_mod = []
            unitsPerMod = self.units // self.clock_numPeriods
            
            def if_true():
                hModule = K.dot(h_tm1[:,s:], self.rec_kernel_c_mod[i]) + K.dot(inputs, self.kernel_c_mod[i]);
                if self.use_bias:
                    hModule = K.bias_add(hModule, self.bias_mod[i])
                if self.recurrent_activation is not None:
                    hModule = self.recurrent_activation(hModule)
                return hModule
            
            def if_false():
              return hModule
            
            for i, period in enumerate(self.clock_periods):
                s = i*unitsPerMod
                e = (i+1)*unitsPerMod
                hModule = h_tm1[:,s:e]
                h_mod.append(tf.cond(K.equal(K.tf.mod(time_step[0][0], period), 0), if_true, if_false))
            hidden = K.concatenate(h_mod)

        else:
            # Update State, all at once, then only use certain updates
            h = K.dot(inputs, self.kernel) + K.dot(h_tm1, self.recurrent_kernel_c*self.cw_mask)
            if self.bias is not None:
                h = K.bias_add(h, self.bias)
            if self.recurrent_activation is not None:
                h = self.recurrent_activation(h)
            
            h = K.switch(K.equal(K.tf.mod(time_step, self.cw_periods), 0), h, h_tm1)
            hidden = h

        # Calculate Output
        output = K.dot(hidden, self.recurrent_kernel_o)
        if self.activation is not None:
            output = self.activation(output)

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                output._uses_learning_phase = True
        return output, [hidden, time_step+1]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'clock_periods' : self.clock_periods}
        base_config = super(CWRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CWRNN(RNN):
    """Clockwork RNN Class
    Based off SimpleRNN from Keras
    Made by Dean Reading 2018

    # Arguments
        units: Positive integer, dimensionality of the output space.
        clock_periods: A list of the clock periods for each 'module'. 
            In the paper, it's [1,2,4,8,16].
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        split_method:
            I implemented the CWRNN via 2 different methods, which should
            (theoretically) be equivalent. The split method updates each module
            1-by-1, and the standard method updates all weights at once. The 
            split method turned out to be slower, so I don't use it by default.
            But, I've included it for evaluation / discussion / validation.
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units, clock_periods,
                 activation='tanh',
                 recurrent_activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 split_method=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            warnings.warn('The `implementation` argument '
                          'in `CWRNN` has been deprecated. '
                          'Please remove it from your layer call.')
        if K.backend() == 'theano':
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = CWRNNCell(units,
                             activation=activation,
                             recurrent_activation=recurrent_activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             recurrent_initializer=recurrent_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             recurrent_regularizer=recurrent_regularizer,
                             bias_regularizer=bias_regularizer,
                             kernel_constraint=kernel_constraint,
                             recurrent_constraint=recurrent_constraint,
                             bias_constraint=bias_constraint,
                             dropout=dropout,
                             recurrent_dropout=recurrent_dropout,
                             clock_periods=clock_periods,
                             split_method=split_method)
        super(CWRNN, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        unroll=unroll,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(CWRNN, self).call(inputs,
                                           mask=mask,
                                           training=training,
                                           initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation
    
    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout
    
    @property
    def clock_periods(self):
        return self.cell.clock_periods
    
    @property
    def split_method(self):
        return self.cell.split_method

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(CWRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


