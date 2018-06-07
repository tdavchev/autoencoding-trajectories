import time
import argparse

import numpy as np
import tensorflow as tf

from time import gmtime, strftime
from sklearn.model_selection import train_test_split

from utils.utils import LoadTrajData
from models.model import Seq2seqModel

def main():
    '''
    Main function. Sets up all arguments
    '''
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--num_units', type=int, default=15,
                        help='size of RNN hidden state')
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=15,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    # Number of epochs parameter
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs')
    # Gradient value at which it should be clipped
    parser.add_argument('--max_gradient_norm', type=float, default=1.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    # Dimension of the embeddings parameter
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--model_path', type=str, default='./save/model',
                        help='Directory to save model to')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or infer')
    parser.add_argument('--content_type', type=str, default='directions',
                        help='locations, directions or 2D-directions')
    parser.add_argument('--tag_type', type=str, default='single',
                        help='Single tag per trajectory sequence or Multiple tags for each step.')
    args = parser.parse_args()
    
    start(args)

def start(args):
    data = load(args)

    model = Seq2seqModel(
        data["data_class"].char2Num,
        args.mode,
        args.num_units,
        args.embed_size,
        args.batch_size,
        args.max_gradient_norm,
        args.learning_rate)

    model.buildModel()

    if args.mode == 'train':
        train(args, model, data)
    else:
        infer(args, model, data)

def load(args):
    # load data
    data = LoadTrajData(contents=args.content_type, tag=args.tag_type)
    x_train, x_test, y_train, y_test = train_test_split(data.input_data, data.target_data, test_size=0.33, random_state=42)

    # Due to shuffle the sequences are not as they are stored in data.seqlen
    # we need the data.seqlen ids for all items in train and test.
    seqlen_idx_train = [[idx for idx, val in enumerate(data.input_data) if val==train_val] for train_val in x_train] # (38, 1)
    seqlen_idx_train = np.reshape(seqlen_idx_train, (len(seqlen_idx_train),)) # (38, )
    seqlen_idx_test =  [[idx for idx, val in enumerate(data.input_data) if val==test_val] for test_val in x_test] # (20, 1)
    seqlen_idx_test = np.reshape(seqlen_idx_test, (len(seqlen_idx_test),)) # (20, )
    # I need to find train points of split only ..

    # Pad data...# needs fixing...
    y_train = data.embedData(y_train, dataType='targets', contents='directions')
    y_test = data.embedData(y_test, dataType='targets', test=True, contents='directions')

    x_train = data.embedData(x_train, contents=args.content_type, dataType='inputs')
    x_test = data.embedData(x_test, contents=args.content_type, test=True)

    return {
        "data_class": data,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "seqlen_idx_train": seqlen_idx_train,
        "seqlen_idx_test": seqlen_idx_test
    }


def train(args, model, data):
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
    saver = tf.train.Saver()
    with tf.Session() as sess:
    # with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logdir/train_'+strftime("%d_%b_%Y_%H_%M_%S", gmtime()))  # create writer
        writer.add_graph(sess.graph)
        print("predicted dictionary: {}".format(data["data_class"].char2Num['targets']))
        print("Starting to learn...")
        num_batches = len(data["x_train"]) // args.batch_size
        for epoch_i in range(args.epochs):
            start_time = time.time()
            for batch_i, (source_batch, target_batch, batch_seqlen, batch_y_seqlen, pos, shuffle) in \
                    enumerate(data["data_class"].batch_data(data["x_train"], data["y_train"], data["seqlen_idx_train"], args.batch_size)):
                
                food = {model.encoder_lengths : batch_seqlen,
                        model.decoder_lengths : batch_y_seqlen,
                        model.inputs          : np.swapaxes(source_batch, 0, 1),
                        # thesse outputs are decoder inputs.. it can doesn't have to consider the last two as they are extra pad + <EOS>
                        model.outputs         : np.swapaxes(target_batch[:, :-1], 0, 1), # should not be padded (I think) append <END> to the end.
                        model.targets         : np.swapaxes(target_batch[:, 1:], 0, 1)}
                _, batch_loss, summary = sess.run([model.update_step, model.train_loss, model.merged], feed_dict=food)
                writer.add_summary(summary, batch_i + num_batches * epoch_i)

            if epoch_i == 0 or epoch_i % 10 == 0:
                print('Batch: {}'.format(batch_i + num_batches * epoch_i))
                print('  minibatch_loss: {}'.format(sess.run(model.train_loss, food)))
                predict_, valid_tar = sess.run([model.dec_predictions, model.valid_targets], food)
                acc = data["data_class"].calculateAccuracy(valid_tar, predict_, batch_y_seqlen)
                print('  accuracy: {}'.format(acc))
                for i, (inp, pred) in enumerate(zip(food[model.targets].T, predict_.T)):
                    end_point = food[model.decoder_lengths][i]
                    split_point = pos[i]
                    print('   sample: {}'.format(i+1))
                    print('      sequence real id         :> {}'.format(shuffle[i]))
                    print('      target start             :> {}'.format(inp[:5]))
                    print('      predicted start          :> {}'.format(pred[:5]))
                    print('      point of split target    :> {}'.format(inp[split_point-5:split_point+5]))
                    print('      point of split predicted :> {}'.format(pred[split_point-5:split_point+5]))
                    print('      target end region        :> {}'.format(inp[end_point-5:end_point+5]))
                    print('      predicted end region     :> {}'.format(pred[end_point-5:end_point+5]))
                    if i >= 2:
                        break
                print()
            print('Epoch: {:3} Loss: {:>6.3f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, time.time() - start_time))
            print("----------------------------------------------------------------------")
            print()
        writer.close()
        saver.save(sess, args.model_path)
        print("model saved to {}".format(args.model_path))
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")

def infer(args, model, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('loading variables...')
        saver.restore(sess, args.model_path)

        print("predicted dictionary: {}".format(data["data_class"].char2Num['targets']))
        print(data["data_class"].char2Num['targets']['<END>'])
        source_batch, target_batch, test_seqlen, test_y_seqlen, pos_test, shuffle = \
            next(data["data_class"].batch_data(data["x_train"], data["y_train"], data["seqlen_idx_train"], args.batch_size))

        dec_input = np.zeros((len(target_batch), len(target_batch[0]))) + data["data_class"].char2Num['targets']['<GO>']
        print("dec_input: ", dec_input.shape)
        print("source_batch: ", source_batch.shape)
        food = {model.encoder_lengths : test_seqlen,
                model.inputs          : np.swapaxes(source_batch, 0, 1),
                model.outputs         : np.swapaxes(dec_input, 0, 1)}

        ans = sess.run(model.translations, feed_dict=food)

        print(len(ans[0]))
        for entry in ans[0]:
            print(entry, end='')

        print()
if __name__ == "__main__":
    main()
