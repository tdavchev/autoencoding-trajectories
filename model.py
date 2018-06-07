import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

from tensorflow.python.layers import core as layers_core

class Seq2seqModel(object):
    def __init__(self, char2Num, mode='train', num_units=15, embed_size=128, batch_size=15, max_gradient_norm=1.0, learning_rate=1e-3):
        self.mode = mode
        self.char2Num = char2Num
        self.num_units = num_units
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate

    def buildModel(self):
        with tf.name_scope('Feed_tensors'):
            # Tensor where we will feed the data into graph
            self.encoder_lengths = tf.placeholder(tf.int32, [None])
            self.decoder_lengths = tf.placeholder(tf.int32, [None])
            self.inputs = tf.placeholder(tf.int32, (None, None), 'inputs')
            self.outputs = tf.placeholder(tf.int32, (None, None), 'output')
            self.targets = tf.placeholder(tf.int32, (None, None), 'targets')

        with tf.name_scope('Embedding_layers'):
            input_embedding = tf.Variable(tf.random_uniform((len(self.char2Num['inputs']), self.embed_size), -1.0, 1.0), name='enc_embedding')
            output_embedding = tf.Variable(tf.random_uniform((len(self.char2Num['targets']), self.embed_size), -1.0, 1.0), name='dec_embedding')
            tf.summary.histogram('input_embedding_var', input_embedding)
            tf.summary.histogram('output_embedding_var', output_embedding)
            # lookup
            # perhaps convert to an "unknown" token the commas
            # and spaces so all get the same embedding
            # Look up embedding:
            #   encoder_inputs  : [max_time, batch_size]
            #   encoder_emb_inp : [max_time, batch_size, embedding_size]
            encoder_emb_inp = tf.nn.embedding_lookup(input_embedding, self.inputs)
            # we need a large amount of training data so we
            # can learn these embeddings from scratch.
            decoder_emb_inp = tf.nn.embedding_lookup(output_embedding, self.outputs)

        with tf.variable_scope('encoding') as encoding_scope:
            encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units)
            # Run Dynamic RNN
            #   encoder_outpus: [max_time, batch_size, num_units]
            #   encoder_state: [batch_size, num_units]
            _, encoder_state = tf.nn.dynamic_rnn(encoder_cell, inputs=encoder_emb_inp, dtype=tf.float32, sequence_length=self.encoder_lengths, time_major=True)

        with tf.variable_scope('decoding') as decoding_scope:
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
            projection_layer = layers_core.Dense(len(self.char2Num['targets']), use_bias=False)

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp, self.decoder_lengths, time_major=True)
                # Decoder
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, encoder_state,
                    output_layer=projection_layer)
                # Dynamic decoding
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...) # what are these dots sorcery..
                logits = decoder_outputs.rnn_output
                self.dec_predictions = tf.argmax(logits, -1, output_type=tf.int32)
                # but then the pads are part of the loss
                # add 0's to the end along the first dimension (0th), do nothing along te rest 2 dims.
                paddings = [[0,tf.shape(self.targets)[0]-tf.shape(logits)[0]], [0,0], [0,0]]
                logits = tf.pad(logits, paddings, 'CONSTANT')
            else:
                maximum_iterations = tf.round(tf.reduce_max(self.encoder_lengths) * 2)
                # Helper
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    output_embedding,
                    start_tokens=tf.fill([tf.shape(self.inputs)[1]], self.char2Num['targets']['<GO>']),
                    end_token=self.char2Num['targets']['<END>'])
                # Decoder
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, encoder_state,
                    output_layer=projection_layer)

                # Dynamic decoding
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=maximum_iterations)
                self.translations = decoder_outputs.sample_id

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            with tf.name_scope("Optimization"):
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.targets, logits=logits)

                target_weights = tf.sequence_mask(
                    tf.shape(self.targets)[0], tf.shape(self.targets)[1], dtype=logits.dtype)

                self.train_loss = (tf.reduce_sum(crossent * target_weights) / self.batch_size)
                tf.summary.scalar('loss', self.train_loss)

                # Calculate and clip gradients
                params = tf.trainable_variables()
                gradients = tf.gradients(self.train_loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

                # Optimization
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

                self.valid_targets = tf.strided_slice(self.targets, [0, 0], [tf.shape(self.dec_predictions)[0], self.batch_size], [1, 1])
                # correct_pred = tf.equal(dec_predictions, valid_targets)

        self.merged = tf.summary.merge_all()
