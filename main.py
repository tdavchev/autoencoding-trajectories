import time
import argparse

import numpy as np
import tensorflow as tf

from time import gmtime, strftime
from sklearn.model_selection import train_test_split

from utils.utils import LoadTrajData
from models.model import Seq2seqModel
from models.model2 import TrajNetwork2D
from models.model3 import TrajNetwork1D
from models.model4 import ActionConditionedTrajNetwork1D

import seaborn as sns
import matplotlib.pyplot as plt

def main():
    '''
    Main function. Sets up all arguments
    '''
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--num_units', type=int, default=64,
                        help='size of RNN hidden state')
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=10,
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
    parser.add_argument('--model_path', type=str, default='./save/action_conditioned_2dinp_1dout/model',  #'./save/model',
                        help='Directory to save model to')
    parser.add_argument('--mode', type=str, default='infer',
                        help='train or infer')
    parser.add_argument('--content_type', type=str, default='2D-directions',
                        help='locations, directions or 2D-directions')
    parser.add_argument('--tag_type', type=str, default='single',
                        help='Single tag per trajectory sequence or Multiple tags for each step.')
    parser.add_argument('--padas', type=str, default='numeric',
                        help='Type of padding (text or numeric).')
    parser.add_argument('--network', type=str, default='ActionConditioned',
                        help='Type of network (Seq2Seq, TrajNetwork1D, TrajNetwork2D or ActionConditioned).')
    parser.add_argument('--input_type', type=str, default='centered_at_start',
                        help='Type of input (basic (x,y) or centered_at_start).')
    parser.add_argument('--target_type', type=str, default='normalized_time',
                        help='Type of target (basic (x,y) or normalized_time).')
    args = parser.parse_args()

    start(args)

def start(args):
    if args.network == 'Seq2Seq':
        data = load(args)
        model = Seq2seqModel(
            data["data_class"].char2Num,
            args.mode,
            args.num_units,
            args.embed_size,
            args.batch_size,
            args.max_gradient_norm,
            args.learning_rate)
        infer = inferSeq2Seq
        train = trainSeq2Seq
    elif args.network == 'TrajNetwork2D':
        # target_type='basic' # it should output 2D Gaussians
        data = load(args)
        model = TrajNetwork2D(
            data["data_class"].char2Num,
            data["data_class"].max_len['inputs'])
        infer = sample2D
        train = train2D
    elif args.network == 'TrajNetwork1D':
        data = load(args)
        model = TrajNetwork1D(
            data["data_class"].char2Num,
            data["data_class"].max_len['inputs'])
        infer = sample1D
        train = train1D
    elif args.network == 'ActionConditioned':
        data = load(args)
        model = ActionConditionedTrajNetwork1D(
            data["data_class"].char2Num,
            data["data_class"].max_len['inputs'])
        infer = sample1DActionCond
        train = train1DActionCond

    model.buildModel()

    if args.mode == 'train':
        train(args, model, data)
    else:
        infer(args, model, data)

def load(args):
    # load data
    data = LoadTrajData(
        input_type=args.input_type,
        target_type=args.target_type,
        contents=args.content_type,
        tag=args.tag_type)
    x_train, x_test, y_train, y_test = train_test_split(
        data.input_data,
        data.target_data,
        test_size=0.2,
        random_state=42)

    # Due to shuffle the sequences are not as they are stored in data.seqlen
    # we need the data.seqlen ids for all items in train and test.
    # perhaps discard the always starting from origin thing
    # at the moment padding data to 0 is an issue since starting at origin is 0...
    seqlen_idx_train = [[idx for idx, val in enumerate(data.input_data) if val==train_val] for train_val in x_train] # (38, 1)
    seqlen_idx_train = np.reshape(seqlen_idx_train, (len(seqlen_idx_train),)) # (38, )
    seqlen_idx_test =  [[idx for idx, val in enumerate(data.input_data) if val==test_val] for test_val in x_test] # (20, 1)
    seqlen_idx_test = np.reshape(seqlen_idx_test, (len(seqlen_idx_test),)) # (20, )
    # I need to find train points of split only ..

    # Pad data...# needs fixing...
    if args.content_type != '2D-directions':
        y_train = data.embedData(y_train, dataType='targets', contents=args.content_type)
        y_test = data.embedData(y_test, dataType='targets', test=True, contents=args.content_type)
        x_train = data.embedData(x_train, idxes=seqlen_idx_train, contents=args.content_type, dataType='inputs', padas=args.padas)
        x_test = data.embedData(x_test, idxes=seqlen_idx_test, contents=args.content_type, test=True, padas=args.padas)
        act_train = []
        act_test = []
    else:
        x_train, act_train = data.embedData(x_train, idxes=seqlen_idx_train, contents=args.content_type, dataType='inputs', padas=args.padas)
        x_test, act_test = data.embedData(x_test, idxes=seqlen_idx_test, contents=args.content_type, test=True, padas=args.padas)

    return {
        "data_class": data,
        "x_train": x_train,
        "x_test": x_test,
        "act_train": act_train,
        "act_test": act_test,
        "y_train": y_train,
        "y_test": y_test,
        "seqlen_idx_train": seqlen_idx_train,
        "seqlen_idx_test": seqlen_idx_test
    }

def sample1DActionCond(args, model, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('loading variables...')
        saver.restore(sess, args.model_path)

        test_data = \
            next(data["data_class"].batch_data(data["x_test"], data["y_test"], data["seqlen_idx_test"], args.batch_size))
        
        i = 0
        print(np.array(test_data["actions"]).shape)
        for act in np.swapaxes(np.reshape(test_data["actions"], (args.batch_size, 1)), 0, 1):
            if i < 1:
                i += 1
                food = {model.input_lengths   : test_data["seqlen"],
                        model.inputs          : test_data["inputs"],
                        model.action_lengths  : test_data["seqlen_a"],
                        model.actions         : np.reshape(act, (args.batch_size, 1)),}
            
        o_mux, o_sx = sess.run([model.mux, model.sx], feed_dict=food)
        print("actions", test_data["actions"][0])

        ans = []
        br = 0
        for br, ele in enumerate(test_data["shuffle_ids"]):
            if ele == 0:
                idx = br
        
        for i in range(1000):
            ans.append(sample_gaussian_1d(o_mux[idx][0], o_sx[idx][0], 1))

        print(test_data["shuffle_ids"])

        sns.set(color_codes=True)
        sns.distplot(ans, label='prediction')
        sns.rugplot([test_data["targets"][idx]], color='indianred', label='target')
        plt.title("entry id: "+ str(test_data["shuffle_ids"][idx]))
        plt.ylabel('mass')
        plt.xlabel('time')
        plt.xlim((0, 1))
        plt.legend()

        plt.show()


def train1DActionCond(args, model, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
    # with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logdir/train_'+strftime("%d_%b_%Y_%H_%M_%S", gmtime()))  # create writer
        writer.add_graph(sess.graph)
        print("Starting to learn...")
        print("----------------------------------------------------------------------")
        num_batches = len(data["x_train"]) // args.batch_size
        for epoch_i in range(args.epochs):
            start_time = time.time()
            for batch_i, batch_data in \
                    enumerate(data["data_class"].batch_data(data["x_train"], data["y_train"], data["seqlen_idx_train"], args.batch_size, data["act_train"])):
                i = 0
                for act in np.swapaxes(batch_data["actions"], 0, 1):
                    if i < 1:
                        food = {model.input_lengths   : batch_data["seqlen"],
                                model.inputs          : batch_data["inputs"],
                                model.action_lengths  : batch_data["seqlen_a"],
                                model.actions         : np.reshape(act, (args.batch_size, 1)),
                                model.targets         : np.array(batch_data["targets"]).reshape((len(batch_data["targets"]), 1))}
                        _, batch_loss, summary = sess.run([model.update_step, model.train_loss, model.merged], feed_dict=food)
                        writer.add_summary(summary, batch_i + num_batches * epoch_i)    
                        i+= 1
            print('Epoch: {:3} Loss: {:>6.3f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, time.time() - start_time))
            print("----------------------------------------------------------------------")
            print()
        writer.close()
        saver.save(sess, args.model_path)
        print("model saved to {}".format(args.model_path))
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")

def sample_gaussian_1d(mean, cov, samples):
    '''
    Function to sample a point from a given 2D normal distribution
    params:
    mux : mean of the distribution in x
    muy : mean of the distribution in y
    sx : std dev of the distribution in x
    sy : std dev of the distribution in y
    rho : Correlation factor of the distribution
    '''
    x = np.random.normal(mean, cov, samples)
    return x

def sample1D(args, model, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('loading variables...')
        saver.restore(sess, args.model_path)

        test_data = \
            next(data["data_class"].batch_data(data["x_test"], data["y_test"], data["seqlen_idx_test"], args.batch_size))

        food = {model.input_lengths   : test_data["seqlen"],
                model.inputs          : test_data["inputs"]}
        
        o_mux, o_sx = sess.run([model.mux, model.sx], feed_dict=food)
        print("actions", test_data["actions"][0])

        for br, ele in enumerate(test_data["shuffle_ids"]):
            if ele == 0:
                idx = br
        
        ans = []
        for i in range(1000):
            ans.append(sample_gaussian_1d(o_mux[idx][0], o_sx[idx][0], 1))

        print(test_data["shuffle_ids"])

        sns.set(color_codes=True)
        sns.distplot(ans, label='prediction')
        sns.rugplot([test_data["targets"][idx]], color='indianred', label='target')
        plt.title("entry id: "+ str(test_data["shuffle_ids"][idx]))
        plt.ylabel('mass')
        plt.xlabel('time')
        plt.xlim((0, 1))
        plt.legend()

        plt.show()

def train1D(args, model, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
    # with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logdir/train_'+strftime("%d_%b_%Y_%H_%M_%S", gmtime()))  # create writer
        writer.add_graph(sess.graph)
        print("Starting to learn...")
        print("----------------------------------------------------------------------")
        num_batches = len(data["x_train"]) // args.batch_size
        for epoch_i in range(args.epochs):
            start_time = time.time()
            for batch_i, batch_data in \
                    enumerate(data["data_class"].batch_data(data["x_train"], data["y_train"], data["seqlen_idx_train"], args.batch_size, data["act_train"])):
                i = 0
                for act in np.swapaxes(batch_data["actions"], 0, 1):
                    if i < 1:
                        food = {model.input_lengths   : batch_data["seqlen"],
                                model.inputs          : batch_data["inputs"],
                                model.targets         : np.array(batch_data["targets"]).reshape((len(batch_data["targets"]), 1))}
                        _, batch_loss, summary = sess.run([model.update_step, model.train_loss, model.merged], feed_dict=food)
                        writer.add_summary(summary, batch_i + num_batches * epoch_i)    
                        i+= 1
            print('Epoch: {:3} Loss: {:>6.3f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, time.time() - start_time))
            print("----------------------------------------------------------------------")
            print()
        writer.close()
        saver.save(sess, args.model_path)
        print("model saved to {}".format(args.model_path))
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")


def sample_gaussian_2d(mux, muy, sx, sy, rho):
    '''
    Function to sample a point from a given 2D normal distribution
    params:
    mux : mean of the distribution in x
    muy : mean of the distribution in y
    sx : std dev of the distribution in x
    sy : std dev of the distribution in y
    rho : Correlation factor of the distribution
    '''
    # Extract mean
    mean = [mux, muy]
    # Extract covariance matrix
    cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
    # Sample a point from the multivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def gaussian_2d(x, y, x0, y0, xsig, ysig, corr):
    return np.exp((-0.5*(1-corr**2)) * (((x-x0) / xsig)**2 + ((y-y0) / ysig)**2 - (2*corr*(x-x0)*(y-y0)/xsig*ysig))) / (2*np.pi*xsig*ysig*np.sqrt(1-corr**2))

def sample2D(args, model, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('loading variables...')
        saver.restore(sess, args.model_path)

        test_data = \
            next(data["data_class"].batch_data(data["x_test"], data["y_test"], data["seqlen_idx_test"], args.batch_size))

        food = {model.input_lengths   : test_data["seqlen"],
                model.inputs          : test_data["inputs"]}
        o_mux, o_muy, o_sx, o_sy, o_corr = sess.run([model.mux, model.muy, model.sx, model.sy, model.corr], feed_dict=food)
        next_x, next_y = sample_gaussian_2d(o_mux[0][0], o_muy[0][0], o_sx[0][0], o_sy[0][0], o_corr[0][0])
        
        print("next_x, next_y :", (next_x, next_y))
        print("target: ", test_data["targets"][0])
        print("o_mux, o_muy :", (o_mux[0][0], o_mux[0][0]))
        print("o_sx, o_sy :", (o_sx[0][0], o_sy[0][0]))
        print("o_corr :", o_corr[0][0])

        print("actions", test_data["actions"][0])

        ans = []
        for p in range(200):
            ans.append(sample_gaussian_2d(o_mux[0][0], o_muy[0][0], o_sx[0][0], o_sy[0][0], o_corr[0][0]))

        ans = np.array(ans)
        x = np.reshape(ans[:, 0], (200,)).tolist()
        y = np.reshape(ans[:, 1], (200,)).tolist()

        X, Y = np.meshgrid(x, y)

        Z = gaussian_2d(X, Y, o_mux[0][0], o_muy[0][0], o_sx[0][0], o_sy[0][0], o_corr[0][0])

        plt.contourf(X, Y, Z, cmap='Blues')
        plt.plot(test_data["inputs"][0][:test_data["seqlen"][0], 0], test_data["inputs"][0][:test_data["seqlen"][0], 1], 'r')
        # plt.plot(ans[:, 0], ans[:, 1])
        plt.show()

def train2D(args, model, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
    # with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logdir/train_'+strftime("%d_%b_%Y_%H_%M_%S", gmtime()))  # create writer
        writer.add_graph(sess.graph)
        # print("predicted dictionary: {}".format(data["data_class"].char2Num['targets']))
        print("Starting to learn...")
        print("----------------------------------------------------------------------")
        num_batches = len(data["x_train"]) // args.batch_size
        for epoch_i in range(args.epochs):
            start_time = time.time()
            # {
            #     "input": data[start:start+batch_size],
            #     "actions": actions[start:start+batch_size],
            #     "targets": labels[start:start+batch_size],
            #     "seqlen": seqlen[start:start+batch_size],
            #     "seqlen_a": a_seqlen[start:start+batch_size],
            #     "seqlen_y": y_seqlen[start:start+batch_size],
            #     "points_of_split": p_of_split[start:start+batch_size],
            #     "shuffle_ids": shuffle[start:start+batch_size]
            # }
            for batch_i, batch_data in \
                    enumerate(data["data_class"].batch_data(data["x_train"], data["y_train"], data["seqlen_idx_train"], args.batch_size, data["act_train"])):
                i = 0
                # new_out =[]
                # for b_no, elem in enumerate(batch_data["inputs"]):
                #     # print(np.array(elem[:batch_data["seqlen"][b_no]]).shape)
                #     new_out.append(elem[:batch_data["seqlen"][b_no]])

                # new_out = np.array(new_out)
                for act, targ in zip(np.swapaxes(batch_data["actions"], 0, 1), np.swapaxes(batch_data["targets"], 0, 1)):
                    if i < 1:
                        food = {model.input_lengths   : batch_data["seqlen"],
                                model.inputs          : batch_data["inputs"],
                                model.targets         : targ}
                        _, batch_loss, summary = sess.run([model.update_step, model.train_loss, model.merged], feed_dict=food)
                        writer.add_summary(summary, batch_i + num_batches * epoch_i)
                            
                        # next_x, next_y = sample_gaussian_2d(o_mux[0][0], o_muy[0][0], o_sx[0][0], o_sy[0][0], o_corr[0][0])
                        i+= 1
            print('Epoch: {:3} Loss: {:>6.3f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, time.time() - start_time))
            print("----------------------------------------------------------------------")
            print()
        writer.close()
        saver.save(sess, args.model_path)
        print("model saved to {}".format(args.model_path))
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")

def trainSeq2Seq(args, model, data):
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
        print("----------------------------------------------------------------------")
        num_batches = len(data["x_train"]) // args.batch_size
        for epoch_i in range(args.epochs):
            start_time = time.time()
            for batch_i, batch_data in \
                    enumerate(data["data_class"].batch_data(data["x_train"], data["y_train"], data["seqlen_idx_train"], args.batch_size)):
                
                food = {model.encoder_lengths : batch_data["seqlen"],
                        model.decoder_lengths : batch_data["seqlen_y"],
                        model.inputs          : np.swapaxes(batch_data["inputs"], 0, 1),
                        # thesse outputs are decoder inputs.. it can doesn't have to consider the last two as they are extra pad + <EOS>
                        model.outputs         : np.swapaxes(np.array(batch_data["targets"])[:, :-1], 0, 1), # should not be padded (I think) append <END> to the end.
                        model.targets         : np.swapaxes(np.array(batch_data["targets"])[:, 1:], 0, 1)}
                _, batch_loss, summary = sess.run([model.update_step, model.train_loss, model.merged], feed_dict=food)
                writer.add_summary(summary, batch_i + num_batches * epoch_i)

            print('Epoch: {:3} Loss: {:>6.3f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, time.time() - start_time))

            if epoch_i == 0 or epoch_i % 10 == 0:
                print('Batch: {}'.format(batch_i + num_batches * epoch_i))
                print('  minibatch_loss: {}'.format(sess.run(model.train_loss, food)))
                predict_, valid_tar = sess.run([model.dec_predictions, model.valid_targets], food)
                acc = data["data_class"].calculateAccuracy(np.swapaxes(valid_tar, 0, 1), np.swapaxes(predict_, 0, 1), batch_data["seqlen_y"])
                print('  accuracy: {}'.format(acc))
                for i, (inp, pred) in enumerate(zip(food[model.targets].T, predict_.T)):
                    end_point = food[model.decoder_lengths][i]
                    split_point = batch_data["points_of_split"][i]
                    print('   sample: {}'.format(i+1))
                    print('      sequence real id         :> {}'.format(batch_data["shuffle_ids"][i]))
                    print('      target start             :> {}'.format(inp[:5]))
                    print('      predicted start          :> {}'.format(pred[:5]))
                    print('      point of split target    :> {}'.format(inp[split_point-5:split_point+5]))
                    print('      point of split predicted :> {}'.format(pred[split_point-5:split_point+5]))
                    print('      target end region        :> {}'.format(inp[end_point-5:end_point+5]))
                    print('      predicted end region     :> {}'.format(pred[end_point-5:end_point+5]))
                    if i >= 2:
                        break
                print()
            print("----------------------------------------------------------------------")
            print()
        writer.close()
        saver.save(sess, args.model_path)
        print("model saved to {}".format(args.model_path))
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")

def inferSeq2Seq(args, model, data):
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
        ans = []
        # infer separately otherwise every output will have the same size
        # the inference helper is run until all sequences output end statement.
        for i in range(target_batch.shape[0]):
            food = {model.encoder_lengths : [test_seqlen[i]],
                    model.inputs          : np.reshape(source_batch[i, :], (source_batch[i, :].shape[0], 1)),#np.swapaxes(source_batch, 0, 1),
                    model.outputs         : np.reshape(dec_input[i, :], (dec_input[i, :].shape[0], 1))}#np.swapaxes(dec_input, 0, 1)}

            translated = sess.run(model.translations, feed_dict=food)
            ans.append(translated[0])

        acc = data["data_class"].calculateAccuracy(target_batch[:, 1:], ans, test_y_seqlen)

        for idx in range(len(ans)):
            print("Predicted: ", end='')
            for entry in ans[idx]:
                print(entry, end='')

            print()
            print("target: ", target_batch[idx][1:])

            print()
            print("------------")

        print('Test accuracy: {:>6.3f}'.format(acc))

if __name__ == "__main__":
    main()
