import random
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

from tensorflow.python.layers import core as layers_core

from utils import LoadTrajData
from time import gmtime, strftime
from sklearn.model_selection import train_test_split

BATCH_SIZE = 15
NUM_UNITS = 15
EMBED_SIZE = 128
EPOCHS = 40#600
MODEL_PATH = './models/model_actions_samecell'
learning_rate = 1e-3
max_gradient_norm = 5.0

# load data
content_type = 'directions'
data = LoadTrajData(contents=content_type)
X_train, X_test, y_train, y_test = train_test_split(data.input_data, data.target_data, test_size=0.33, random_state=42)

# Due to shuffle the sequences are not as they are stored in data.seqlen
# we need the data.seqlen ids for all items in train and test.
seqlen_idx_train = [[idx for idx, val in enumerate(data.input_data) if val==train_val] for train_val in X_train] # (38, 1)
seqlen_idx_train = np.reshape(seqlen_idx_train, (len(seqlen_idx_train),)) # (38, )
seqlen_idx_test =  [[idx for idx, val in enumerate(data.input_data) if val==test_val] for test_val in X_test] # (20, 1)
seqlen_idx_test = np.reshape(seqlen_idx_test, (len(seqlen_idx_test),)) # (20, )
# I need to find train points of split only ..

# Pad data...# needs fixing...
y_train = data.embedData(y_train, dataType='targets', contents='directions')
y_test = data.embedData(y_test, dataType='targets', test=True, contents='directions')

pad_x_train = data.embedData(X_train, contents=content_type, dataType='inputs')
pad_x_test = data.embedData(X_test, contents=content_type, test=True)

SEQUENCE_LENGTH = [len(pad_x_train[0]), len(y_train[0]) - 1]

with tf.name_scope('Feed_tensors'):
    # Tensor where we will feed the data into graph
    encoder_lengths = tf.placeholder(tf.int32, [None])
    decoder_lengths = tf.placeholder(tf.int32, [None])
    inputs = tf.placeholder(tf.int32, (None, None), 'inputs')
    outputs = tf.placeholder(tf.int32, (None, None), 'output')
    targets = tf.placeholder(tf.int32, (None, None), 'targets')

with tf.name_scope('Embedding_layers'):
    input_embedding = tf.Variable(tf.random_uniform((len(data.char2Num['inputs']), EMBED_SIZE), -1.0, 1.0), name='enc_embedding')
    output_embedding = tf.Variable(tf.random_uniform((len(data.char2Num['targets']), EMBED_SIZE), -1.0, 1.0), name='dec_embedding')
    tf.summary.histogram('input_embedding_var', input_embedding)
    tf.summary.histogram('output_embedding_var', output_embedding)
    # lookup
    # perhaps convert to an "unknown" token the commas
    # and spaces so all get the same embedding
    # Look up embedding:
    #   encoder_inputs  : [max_time, batch_size]
    #   encoder_emb_inp : [max_time, batch_size, embedding_size]
    encoder_emb_inp = tf.nn.embedding_lookup(input_embedding, inputs)
    # we need a large amount of training data so we
    # can learn these embeddings from scratch.
    decoder_emb_inp = tf.nn.embedding_lookup(output_embedding, outputs)

# # use 1 cell for both enc and dec !!!
# (someone said it works better for small quantities data)
# with tf.variable_scope('cell') as cell_scope:
#     cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS)

with tf.variable_scope('encoding') as encoding_scope:
    encoder_cell = tf.contrib.rnn.BasicLSTMCell(NUM_UNITS)
    # Run Dynamic RNN
    #   encoder_outpus: [max_time, batch_size, num_units]
    #   encoder_state: [batch_size, num_units]
    _, encoder_state = tf.nn.dynamic_rnn(encoder_cell, inputs=encoder_emb_inp, dtype=tf.float32, sequence_length=encoder_lengths, time_major=True)

with tf.variable_scope('decoding') as decoding_scope:
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS)
    # Helper
    helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_emb_inp, decoder_lengths, time_major=True)

    projection_layer = layers_core.Dense(
        len(data.char2Num['targets']), use_bias=False)
    # Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, encoder_state,
        output_layer=projection_layer)
    # Dynamic decoding
    decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...) # what are these dots sorcery..
    logits = decoder_outputs.rnn_output

dec_predictions = tf.argmax(logits, -1, output_type=tf.int32)

# but then the pads are part of the loss
# add 0's to the end along the first dimension (0th), do nothing along te rest 2 dims.
paddings = [[0,SEQUENCE_LENGTH[1]-tf.shape(logits)[0]], [0,0], [0,0]]
logits = tf.pad(logits, paddings, 'CONSTANT')


with tf.name_scope("Optimization"):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits)

    target_weights = tf.sequence_mask(
        tf.shape(targets)[0], tf.shape(targets)[1], dtype=logits.dtype)

    train_loss = (tf.reduce_sum(crossent * target_weights) /
        BATCH_SIZE)
    tf.summary.scalar('loss', train_loss)

    # Calculate and clip gradients
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

    # Optimization
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

    valid_targets = tf.strided_slice(targets, [0, 0], [tf.shape(dec_predictions)[0], BATCH_SIZE], [1, 1])
    # correct_pred = tf.equal(dec_predictions, valid_targets)

merged = tf.summary.merge_all()
# session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
saver = tf.train.Saver()

if __name__ == "__main__":
    # with tf.Session(config=session_config) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logdir/train_'+strftime("%d_%b_%Y_%H_%M_%S", gmtime()))  # create writer
        writer.add_graph(sess.graph)
        print("predicted dictionary: {}".format(data.char2Num['targets']))
        print("Starting to learn...")
        num_batches = len(X_train) // BATCH_SIZE
        for epoch_i in range(EPOCHS):
            start_time = time.time()
            for batch_i, (source_batch, target_batch, batch_seqlen, batch_y_seqlen, pos) in enumerate(data.batch_data(pad_x_train, y_train, seqlen_idx_train, BATCH_SIZE)):
                # time major = True (copying NMT tutorial)
                # (a) Encoder inputs (encoder lengths = [3, 2]):
                #   a b c EOS
                #   d e EOS EOS
                # (b) Decoder inputs (decoder lengths = [4, 3]):
                #   GO 1 2 3
                #   GO 4 5 EOS
                # (c) Decoder outputs (shift-by-1 of decoder inputs):
                #   1 2 3 EOS
                #   4 5 EOS EOS
                # (the first EOS is part of the loss)
                # seqlen for targets has + 1 for <GO> symbol...does it have to? I think so.
                food = {encoder_lengths : batch_seqlen,
                        decoder_lengths : batch_y_seqlen,
                        inputs          : np.swapaxes(source_batch, 0, 1),
                        # thesse outputs are decoder inputs.. it can doesn't have to consider the last two as they are extra pad + <EOS>
                        outputs         : np.swapaxes(target_batch[:, :-1], 0, 1), # should not be padded (I think) append <END> to the end.
                        targets         : np.swapaxes(target_batch[:, 1:], 0, 1)}
                _, batch_loss, summary = sess.run([update_step, train_loss, merged], feed_dict=food)
                writer.add_summary(summary, batch_i + num_batches * epoch_i)

            if epoch_i == 0 or epoch_i % 10 == 0:
                print('Batch: {}'.format(batch_i + num_batches * epoch_i))
                print('  minibatch_loss: {}'.format(sess.run(train_loss, food)))
                predict_, valid_tar = sess.run([dec_predictions, valid_targets], food)
                acc = data.calculateAccuracy(valid_tar, predict_, batch_y_seqlen)
                print('  accuracy: {}'.format(acc))
                for i, (inp, pred) in enumerate(zip(food[targets].T, predict_.T)):
                    end_point = food[decoder_lengths][i]
                    split_point = pos[i]
                    print('   sample: {}'.format(i+1))
                    print('      target start             : >{}'.format(inp[:5]))
                    print('      predicted start          : >{}'.format(pred[:5]))
                    print('      point of split target    : >{}'.format(inp[split_point-5:split_point+5]))
                    print('      point of split predicted : >{}'.format(pred[split_point-5:split_point+5]))
                    print('      target end region        : >{}'.format(inp[end_point-5:end_point+5]))
                    print('      predicted end region     : >{}'.format(pred[end_point-5:end_point+5]))
                    if i >= 2:
                        break
                print()
                # acc = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
            print('Epoch: {:3} Loss: {:>6.3f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, time.time() - start_time))
            print("----------------------------------------------------------------------")
            print()
        writer.close()
        saver.save(sess, MODEL_PATH)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")

    with tf.Session() as sess:
        print('loading variables...')
        saver.restore(sess, MODEL_PATH)

        source_batch, target_batch, test_seqlen, test_y_seqlen = next(data.batch_data(pad_x_test, y_test, seqlen_idx_test, BATCH_SIZE))
        dec_input = np.zeros((len(target_batch), len(target_batch[0]))) + data.char2Num['targets']['<GO>']

        print("Evaluating on test set of size: ", len(source_batch))
        batch_logits, acc = sess.run([logits, accuracy],
                    feed_dict = {
                        inputs: source_batch,
                        encoder_lengths: test_seqlen,
                        decoder_lengths: test_y_seqlen,
                        outputs: dec_input[:,:-1],
                        targets: target_batch[:,:-1]})

        print('Accuracy on test set is: {:>6.3f}'.format(acc))

        # Evaluate result:
        num2char = data.numToChar(data.char2Num['targets'])
        for ele in batch_logits.argmax(axis=-1)[3]:
            if num2char[ele] != '<PAD>':
                print(num2char[ele]+"->", end='')
            # print(data.numToChar['targets'][ele])
        print()
        print("----------------------------------------------------")
        for ele in target_batch[3]:
            if num2char[ele] != '<PAD>':
                print(num2char[ele]+"->", end='')
        print()
        print("target ->", target_batch[0])
        print("output ->", batch_logits.argmax(axis=-1)[0])
