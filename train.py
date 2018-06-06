import random
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

from utils import LoadTrajData
from time import gmtime, strftime
from sklearn.model_selection import train_test_split

BATCH_SIZE = 15
NUM_UNITS = 15
EMBED_SIZE = 128
EPOCHS = 40#600
MODEL_PATH = './models/model_actions_samecell'

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

# Pad data..
y_train = data.embedData(y_train, dataType='targets', contents='directions')
y_test = data.embedData(y_test, dataType='targets', test=True, contents='directions')

if content_type != '2D-directions':
    pad_x_train = data.embedData(X_train, contents=content_type, dataType='inputs')
    pad_x_test = data.embedData(X_test, contents=content_type, test=True)
    SEQUENCE_LENGTH = [len(pad_x_train[0]), len(y_train[0])-1]

with tf.name_scope('Feed_tensors'):
    # Tensor where we will feed the data into graph
    encoder_lengths = tf.placeholder(tf.int32, [None])
    decoder_lengths = tf.placeholder(tf.int32, [None])
    inputs = tf.placeholder(tf.int32, (None, SEQUENCE_LENGTH[0]), 'inputs')
    outputs = tf.placeholder(tf.int32, (None, None), 'output')
    targets = tf.placeholder(tf.int32, (None, None), 'targets')

with tf.name_scope('Embedding_layers'):
    input_embedding = tf.Variable(tf.random_uniform((len(data.char2Num['inputs']), EMBED_SIZE), -1.0, 1.0), name='enc_embedding')
    output_embedding = tf.Variable(tf.random_uniform((len(data.char2Num['targets']), EMBED_SIZE), -1.0, 1.0), name='dec_embedding')
    tf.summary.histogram('input_embedding_var', input_embedding)
    tf.summary.histogram('output_embedding_var', output_embedding)
    # lookup
    position_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
    decoder_emb_inp = tf.nn.embedding_lookup(output_embedding, outputs)

# use 1 cell for both enc and dec !!!
with tf.variable_scope('cell') as cell_scope:
    cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS)

with tf.variable_scope('encoding') as encoding_scope:
    # lstm_enc = tf.contrib.rnn.BasicLSTMCell(NUM_UNITS)
    _, encoder_state = tf.nn.dynamic_rnn(cell, inputs=position_input_embed, dtype=tf.float32, sequence_length=encoder_lengths)

with tf.variable_scope('decoding') as decoding_scope:
    dec_outputs, _ = tf.nn.dynamic_rnn(cell, decoder_emb_inp, initial_state=encoder_state)

# connect outputs to
logits = tf.contrib.layers.fully_connected(
    dec_outputs, num_outputs=len(data.char2Num['targets']), activation_fn=None)

# paddings = [[0,0],[0,SEQUENCE_LENGTH[1]-tf.shape(logits)[1]], [0,0]]
# logits = tf.pad(logits, paddings, 'CONSTANT')

with tf.name_scope("Optimization"):
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([BATCH_SIZE, tf.shape(outputs)[1]]))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    correct_pred = tf.equal(tf.argmax(logits,-1, output_type=tf.int32), targets)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
# session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
saver = tf.train.Saver()

if __name__ == "__main__":
    # with tf.Session(config=session_config) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logdir/train_'+strftime("%d_%b_%Y_%H_%M_%S", gmtime()))  # create writer
        writer.add_graph(sess.graph)
        print("Starting to learn...")
        num_batches = len(X_train) // BATCH_SIZE
        for epoch_i in range(EPOCHS):
            start_time = time.time()
            for batch_i, (source_batch, target_batch, batch_seqlen, batch_y_seqlen) in enumerate(data.batch_data(pad_x_train, y_train, seqlen_idx_train, BATCH_SIZE)):
                _, batch_loss, batch_logits, acc, summary = sess.run([optimizer, loss, logits, accuracy, merged],
                    feed_dict={ inputs: source_batch,
                                encoder_lengths: batch_seqlen,
                                decoder_lengths: batch_y_seqlen,
                                outputs: target_batch[:, :-1],
                                targets: target_batch[:, 1:]})
                writer.add_summary(summary, batch_i + num_batches * epoch_i)

            # acc = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
            print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, acc, time.time() - start_time))
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
