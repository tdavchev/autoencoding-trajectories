import numpy as np

import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

from tensorflow.python.layers import core as layers_core

class ActionConditionedTrajNetwork1D(object):
    def __init__(self, char2Num, seq_length, mode='train', num_units=128, embed_size=128, batch_size=15, max_gradient_norm=1.0, learning_rate=1e-3):
        self.mode = mode
        self.char2Num = char2Num
        self.num_units = num_units
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        # Output size is the set of parameters (mu, sigma, corr)
        self.output_size = 2

    def buildEmbeddings(self):
        self.action_embedding = tf.Variable(tf.random_uniform((len(self.char2Num['actions']), self.embed_size), -1.0, 1.0), name='action_embedding')
        tf.summary.histogram('input_embedding_var', self.action_embedding)
        # lookup
        # perhaps convert to an "unknown" token the commas
        # and spaces so all get the same embedding
        # Look up embedding:
        #   encoder_inputs  : [max_time, batch_size]
        #   encoder_emb_inp : [max_time, batch_size, embedding_size]
        embedded_actions = tf.nn.embedding_lookup(self.action_embedding, self.actions)
        # we need a large amount of training data so we
        # can learn these embeddings from scratch.

        return embedded_actions

    def buildModel(self):
        with tf.name_scope('Feed_tensors'):
            # Tensor where we will feed the data into graph
            self.input_lengths = tf.placeholder(tf.int32, [None])
            self.action_lengths = tf.placeholder(tf.int32, [None])
            self.inputs = tf.placeholder(tf.float32, [None, self.seq_length, 2], 'inputs') # sequence length, batch, 2 or reversed..
            self.targets = tf.placeholder(tf.float32, [None, 1], 'targets')
            self.actions = tf.placeholder(tf.int32, [None, 1], 'actopms') # should be grown to 2
        
        with tf.name_scope('Embedding_layers'):
            embedded_actions = self.buildEmbeddings()

        # Embedding for the spatial coordinates
        with tf.variable_scope("coordinate_embedding"):
            embedding_w = tf.get_variable("embedding_w", [2, self.embed_size])
            embedding_b = tf.get_variable("embedding_b", [self.embed_size])

        with tf.variable_scope("tt"):
            tt_w = tf.get_variable("tt_w", [self.num_units*self.seq_length, self.num_units], initializer=tf.truncated_normal_initializer(stddev=0.01), trainable=True)
            tt_b = tf.get_variable("tt_b", [self.num_units], initializer=tf.constant_initializer(0.01), trainable=True)

        # Output linear layer
        with tf.variable_scope("rnnlm"):
            output_w = tf.get_variable("output_w", [self.num_units, self.output_size], initializer=tf.truncated_normal_initializer(stddev=0.01), trainable=True)
            output_b = tf.get_variable("output_b", [self.output_size], initializer=tf.constant_initializer(0.01), trainable=True)

        # Split inputs according to sequences.
        inputs = tf.split(self.inputs, self.seq_length, 1) # none - sequence length ..
        # Get a list of 2D tensors. Each of size numPoints x 2
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # Embed the input spatial points into the embedding space
        embedded_inputs = []
        for x in inputs:
            # Each x is a 2D tensor of size numPoints x 2
            # Embedding layer
            embedded_x = tf.nn.relu(tf.add(tf.matmul(x, embedding_w), embedding_b))
            embedded_inputs.append(embedded_x)

        embedded_inputs = tf.stack(embedded_inputs, 1)

        action_conditioned_inputs = tf.subtract(embedded_inputs, embedded_actions)

        traj_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units, state_is_tuple=False)
        oot, traj_state = tf.nn.dynamic_rnn(traj_cell, inputs=embedded_inputs, dtype=tf.float32, sequence_length=self.input_lengths, time_major=False)
        
        oot = tf.reshape(oot, [tf.shape(oot)[0], -1])
        oot = tf.nn.xw_plus_b(oot, tt_w, tt_b)
        output = tf.nn.xw_plus_b(oot, output_w, output_b)
        self.final_state = traj_state

        # x_data = tf.split(self.targets, 1, -1)
        def tf_1d_normal(x, mux, sx):
            '''
            Function that implements the PDF of a 2D normal distribution
            params:
            x : input x points
            y : input y points
            mux : mean of the distribution in x
            muy : mean of the distribution in y
            sx : std dev of the distribution in x
            sy : std dev of the distribution in y
            rho : Correlation factor of the distribution
            '''
            # eq 3 in the paper
            # and eq 24 & 25 in Graves (2013)
            # Calculate (x - mux) and (y-muy)
            normx = tf.subtract(x, mux)
            # Calculate sx*sy
            # Calculate the exponential factor
            z = tf.square(tf.div(normx, sx))
            # negRho = 1 - tf.square(rho)
            # Numerator
            result = tf.exp(-z)
            # Normalization constant
            denom = tf.sqrt(2 * np.pi) * sx
            # Final PDF calculation
            result = tf.div(result, denom)
            self.result = result
            return result

        # Important difference between loss func of Social LSTM and Graves (2013)
        # is that it is evaluated over all time steps in the latter whereas it is
        # done from t_obs+1 to t_pred in the former
        def get_lossfunc(z_mux, z_sx, x_data):
            '''
            Function to calculate given a 2D distribution over x and y, and target data
            of observed x and y points
            params:
            z_mux : mean of the distribution in x
            z_muy : mean of the distribution in y
            z_sx : std dev of the distribution in x
            z_sy : std dev of the distribution in y
            z_rho : Correlation factor of the distribution
            x_data : target x points
            y_data : target y points
            '''
            # step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

            # Calculate the PDF of the data w.r.t to the distribution
            result0 = tf_1d_normal(x_data, z_mux, z_sx)
            # result0_2 = tf_2d_normal(tf.add(x_data, step), y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
            # result0_3 = tf_2d_normal(x_data, tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
            # result0_4 = tf_2d_normal(tf.add(x_data, step), tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)

            # result0 = tf.div(tf.add(tf.add(tf.add(result0_1, result0_2), result0_3), result0_4), tf.constant(4.0, dtype=tf.float32, shape=(1, 1)))
            # result0 = tf.mul(tf.mul(result0, step), step)

            # For numerical stability purposes
            epsilon = 1e-20

            # TODO: (resolve) I don't think we need this as we don't have the inner
            # summation
            # result1 = tf.reduce_sum(result0, 1, keep_dims=True)
            # Apply the log operation
            result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

            # TODO: For now, implementing loss func over all time-steps
            # Sum up all log probabilities for each data point
            return tf.reduce_sum(result1)

        def get_coef(output):
            # eq 20 -> 22 of Graves (2013)
            # TODO : (resolve) Does Social LSTM paper do this as well?
            # the paper says otherwise but this is essential as we cannot
            # have negative standard deviation and correlation needs to be between
            # -1 and 1

            z = output
            # Split the output into 5 parts corresponding to means, std devs and corr
            z_mux, z_sx, = tf.split(z, 2, -1)

            # The output must be exponentiated for the std devs
            z_sx = tf.exp(z_sx)

            return [z_mux, z_sx]

        [o_mux, o_sx] = get_coef(output)
        self.output = output

        # Store the predicted outputs
        self.mux = o_mux
        self.sx = o_sx

        # Compute the loss function
        lossfunc = get_lossfunc(o_mux, o_sx, self.targets)

        # target_weights = tf.sequence_mask(
        #             tf.shape(self.targets)[0], tf.shape(self.targets)[1], dtype=logits.dtype)

        self.train_loss = tf.div(lossfunc, (self.batch_size)) # this should be wrong I need to multiply by the actual sequence and batch size..
        tf.summary.scalar('loss', self.train_loss)
        params = tf.trainable_variables()

        self.gradients = tf.gradients(self.train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, self.max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

        self.merged = tf.summary.merge_all()
