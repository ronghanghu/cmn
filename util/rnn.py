from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

def lstm_layer(name, seq_bottom, const_bottom, output_dim, num_layers=1,
               forget_bias=0.0, apply_dropout=False, keep_prob=0.5,
               concat_output=True, sequence_length=None):
    """
    Similar LSTM layer to the `LSTMLayer` in Caffe
    ----
    Args:
        seq_bottom : the underlying sequence input of size [T, N, D_in], where
            D_in is the input dimension, T is num_steps and N is batch_size.
        const_bottom : the constant bottom concatenated to each time step,
            having shape [N, D_const]. This can be *None*. If it is None,
            then this input is ignored.
        output_dim : the number of hidden units in the LSTM unit and also the
            final output dimension, i.e. D_out.
        num_layers : the number of stacked LSTM layers.
        forget_bias : forget gate bias in LSTM unit.
        apply_dropout, keep_prob: dropout applied to the output of each LSTM
            unit.
        concat_output : if True, concatenate the ouputs (as is done in Caffe)
        sequence_length : (Optional) Tensor of size [N], contains length of each
                          sequence.

    Returns:
        output : a Tensor of [T, N, D_out] (if concat_output is True) or
                 a list of T Tensors of [N, D_out] (if concat_output is False),
                 where D_out is output_dim, T is num_steps and N is batch_size
    """

    # input shape is [T, N, D_in]
    input_shape = seq_bottom.get_shape().as_list()
    # the number of time steps to unroll
    num_steps = input_shape[0]
    # batch size (i.e. N)
    input_dim = np.prod(input_shape[2:])

    # The actual parameter variable names are as follows (`name` is the name
    # variable here, and Cell0, Cell1, ... are num_layers stacked LSTM cells):
    #   `name`/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix
    #   `name`/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias
    #   `name`/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix
    #   `name`/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias
    # where Cell1 is on top of Cell0, taking Cell0's hidden states as inputs.
    #
    # For Cell0, the weight matrix ('BasicLSTMCell/Linear/Matrix') has shape
    # [D_in+D_const+D_out, 4*D_out], and bias has shape [4*D_out].
    # For Cell1, Cell2, ..., the weight matrix ('BasicLSTMCell/Linear/Matrix')
    # has shape [D_out*2, 4*D_out], and bias has shape [4*D_out].
    # In the weight matrix, the first D_in+D_const rows (in Cell0) or D_out rows
    # (in Cell1, Cell2, ...) are bottom input weights, and the rest D_out rows
    # are state weights, i.e. *inputs are before states in weight matrix*
    #
    # The gate order in 4*D_out are i, j (i.e. g), f, o, where
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    # *this gate order is different from the i, f, o, g order in Caffe LSTM*
    #
    # Other details in tensorflow/python/ops/rnn_cell.py
    with tf.variable_scope(name):
        # the basic LSTM cell
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(output_dim, forget_bias,
            state_is_tuple=True)

        # Apply dropout if specified.
        if apply_dropout and keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers,
                                               state_is_tuple=True)
        else:
            cell = lstm_cell

        # Split along time dimension and flatten each component.
        # `inputs` is a list.
        inputs = [tf.reshape(input_, [-1, input_dim])
            for input_ in tf.split(seq_bottom, num_steps, axis=0)]
        # Add constant input to each time step.
        if not const_bottom is None:
            # Flatten const_bottom into shape [N, D_const] and concatenate.
            const_dim = np.prod(const_bottom.get_shape().as_list()[1:])
            const_input_ = tf.reshape(const_bottom, [-1, const_dim])
            inputs = [tf.concat([input_, const_input_], axis=0)
                for input_ in inputs]

        # Create the Recurrent Network and collect `outputs`. `states` are
        # ignored.
        outputs, _ = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32,
                                               sequence_length=sequence_length)
        if concat_output:
            # Concat the outputs into [T, N, D_out].
            outputs = tf.reshape(tf.concat(outputs, axis=0),
                                [num_steps, -1, output_dim])
    return outputs
