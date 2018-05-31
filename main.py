import logging, argparse, random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MD-LSTM')
    parser.add_argument('-a', '--arch', help='NN architecture', type=str, default='mdlstm')
    parser.add_argument('--rnn_type', help='which type of RNN to use', type=str, default='dynamic')
    parser.add_argument('-d', '--data', help='data type', type=str, default='ir')
    parser.add_argument('-f', '--feature', help='ir feature used to generate match matrix',
                        type=str, default='tf_proximity')
    parser.add_argument('-E', '--eval', help='how many epochs to run before evalutaion', type=int, default=0)
    parser.add_argument('-V', '--visualize', help='how many epochs to run before visualization',
                        type=int, default=0)
    parser.add_argument('-e', '--epoch', help='how many epochs to run', type=int, default=15000)
    parser.add_argument('-s', '--save_model_path', help='path to store the model',
                        type=str, default='trained_model/model')
    parser.add_argument('-S', '--save_model_epoch', help='how many epochs to run before save',
                        type=int, default=3000)
    parser.add_argument('-l', '--load_model_path', help='path to load the model from', type=str, default=None)
    parser.add_argument('-D', '--debug', help='whether to use debug log level', action='store_true')
    parser.add_argument('-B', '--tf_summary_path', help='path to save tf summary', type=str, default=None)
    parser.add_argument('--tf_summary', help='how many epochs to run before summarization',
                        type=int, default=0)
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

from time import time
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python import debug as tf_debug

from data_gen import next_batch
from md_lstm import *
from cnn import cnn, CNNVis
from utils import printoptions, tf_jacobian
from ir_match_matrix import UNIFORM_NOISE
SEED = 2018
random.seed(SEED)
np.random.seed(SEED)


def get_variables(sess):
    graph = tf.get_default_graph()
    return [(v.name, v.eval(sess))
            for v in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]


def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


def multi_dimensional_lstm(input_data, rnn_size):
    rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=rnn_size, input_data=input_data, sh=[1, 1])
    return rnn_out


def standard_lstm(input_data, rnn_size):
    b, h, w, c = input_data.get_shape().as_list()
    new_input_data = tf.reshape(input_data, (b, h * w, c))
    rnn_out, _ = dynamic_rnn(tf.contrib.rnn.LSTMCell(rnn_size),
                             inputs=new_input_data,
                             dtype=tf.float32)
    rnn_out = tf.reshape(rnn_out, (b, h, w, rnn_size))
    return rnn_out


def main():
    if args.arch not in {'lstm', 'mdlstm', 'cnn'}:
        raise Exception('not support arch type (should be one of "lstm", "mdlstm", "cnn").')

    # config
    visualization = 'saliency' # kernel, saliency
    learning_rate = 0.001 * 0.5
    anisotropy = False
    distribution = 'power_law'
    mean_match_query_term = 3
    mean_match_count = 5
    batch_size = 256
    h = 5
    w = 10
    channels = 1
    hidden_size = 50
    mean_match_doc_term = max(1, int(mean_match_count * h / mean_match_query_term))
    cnn_arch = [[3, 3, 1, 16], [1, 2, 2, 1], [3, 3, 16, 32], [1, 2, 2, 1], [2, 2, 32, 64], [1, 2, 2, 1]]
    #cnn_arch = [[3, 3, 1, 1], [1, 1, 1, 1]]
    cnn_activation = 'relu'

    # graph
    #grad_debugger = tf_debug.GradientsDebugger()
    #if args.tf_summary:
    #    global_step = tf.train.get_or_create_global_step()
    #    summary_writer = tf.contrib.summary.create_file_writer(args.tf_summary_path, flush_millis=10000)

    tf.set_random_seed(SEED)
    x = tf.placeholder(tf.float32, [None, h, w, channels])
    y = tf.placeholder(tf.float32, [None, h, w, channels])
    x_w = tf.placeholder(tf.float32, [None, h, w])
    bs = tf.shape(x)[0]

    #with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    if args.arch == 'mdlstm':
        print('Using Multi Dimensional LSTM !')
        if args.rnn_type == 'dynamic':
            nn_out, rnn_states = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
        elif args.rnn_type == 'static':
            nn_out, rnn_states = multi_dimensional_rnn_static(rnn_size=hidden_size, input_data=x, sh=[1, 1])
        #debug_rnn_states = grad_debugger.identify_gradient(rnn_states)
    elif args.arch == 'lstm':
        print('Using Standard LSTM !')
        nn_out = standard_lstm(input_data=x, rnn_size=hidden_size)
    elif args.arch == 'cnn':
        print('Using CNN !')
        nn_out = cnn(x, architecture=cnn_arch, activation=cnn_activation)
        nn_out = tf.reshape(nn_out, (bs, int(np.prod(nn_out.get_shape()[1:]))))

    # linear transformation (no activation)
    model_out = slim.fully_connected(inputs=nn_out,
                                     num_outputs=1,
                                     activation_fn=None)

    if args.arch == 'cnn':
        loss = 1e4 * tf.reduce_sum(tf.abs(
            tf.reshape(tf.boolean_mask(y, tf.expand_dims(x_w, axis=-1) > 0), [bs, 1]) - model_out)) / \
               tf.cast(bs, tf.float32)
    else:
        loss = 1e4 * tf.reduce_sum(tf.abs(y - model_out) * tf.expand_dims(x_w, axis=-1)) / \
               tf.reduce_sum(x_w)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    grad_update = optimizer.minimize(loss)
    if args.arch == 'cnn':
        saliency = tf.gradients(-loss, x)
    else:
        used_model_out = model_out * tf.expand_dims(x_w, axis=-1)
        saliency = tf.gradients(used_model_out, x)
        if args.rnn_type == 'static':
            rnn_states_grad = tf.gradients(used_model_out, [s.c for s in rnn_states])
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    #merged_summary = tf.summary.merge_all()
    #merged_summary = tf.contrib.summary.all_summary_ops()

    # session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #if args.tf_summary:
    #    tf.contrib.summary.initialize(graph=sess.graph, session=sess)
    #    train_writer = tf.summary.FileWriter(args.tf_summary_path, sess.graph)
    # train_writer.add_graph(sess.graph)

    # load model or init model
    if type(args.load_model_path) is str:
        logging.info('load model from "{}"'.format(args.load_model_path))
        saver.restore(sess, args.load_model_path)
    else:
        sess.run(init)

    # train model
    epochs = args.epoch
    for i in range(epochs):
        if args.data == 'gau':
            batch = next_batch(args.data, batch_size, h, w, anisotropy)
        elif args.data == 'ir':
            batch = next_batch(args.data, batch_size, h, w,
                               mean_match_query_term=mean_match_query_term,
                               mean_match_doc_term=mean_match_doc_term,
                               dist=distribution)
        st = time()
        batch_x = np.expand_dims(batch[0], axis=3)
        batch_y = np.expand_dims(batch[1], axis=3)
        batch_x_w = batch[2]

        #if args.debug:
        #    print(batch[0][0])
        #    print(batch[1][0])
        #    print(batch[2][0])
        #    input()

        if args.arch == 'lstm' and i == 0:
            print('Shuffling the batch in the height dimension for the standard LSTM.'
                  'Its like having h LSTM on the width axis.')
            perms = np.random.permutation(list(range(w)))
            batch_x = batch_x[:, perms, :, :]
            batch_y = batch_y[:, perms, :, :]
            pass

        loss_val, _ = sess.run([loss, grad_update], feed_dict={x: batch_x, y: batch_y, x_w: batch_x_w})
        # console output
        if i % 50 == 0:
            print('epochs = {0} | loss = {1:.3f} | time {2:.3f}'.format(str(i).zfill(3),
                                                                       loss_val,
                                                                       time() - st))
            #print([v[0] for v in get_variables(sess)])
            #input()
        # save model
        if args.save_model_path and args.save_model_epoch > 0 and i > 0 and \
                (i % args.save_model_epoch == 0 or i == epochs - 1):
            logging.info('save model to "{}" at epochs {}'.format(args.save_model_path, i))
            saver.save(sess, args.save_model_path)
        # visualize model
        if args.visualize and i % args.visualize == 0:
            cnn_vis = CNNVis()
            if visualization == 'kernel' and args.arch == 'cnn':
                conv_kernels, = sess.run([tf.get_collection('conv_kernel')])
                kmax = np.max([np.max(k) for k in conv_kernels])
                kmin = np.min([np.min(k) for k in conv_kernels])
                print('kernal max: {}, min: {}'.format(kmax, kmin))
                kmax = max(abs(kmax), abs(kmin)) or 1
                kmin = -kmax
                cnn_vis.set_max_min(kmax, kmin)
                for i, c in enumerate(tf.get_collection('conv_kernel')):
                    cnn_vis.plot_conv_kernel(conv_kernels[i], c.name)
                    input('press to continue')
            elif visualization == 'saliency':
                saliency_map, = \
                    sess.run(saliency, feed_dict={x: batch_x, y: batch_y, x_w: batch_x_w})
                if args.arch == 'mdlstm':
                    # erase the gradiant at the top-left corner when the corresponding input is zero
                    saliency_mask = np.any(np.abs(batch_x[:, 0, 0, :]) > UNIFORM_NOISE,
                                           axis=1, keepdims=True).astype(np.float32)
                    saliency_map[:, 0, 0, :] = saliency_map[:, 0, 0, :] * saliency_mask
                cnn_vis.plot_saliency_map(batch_x, saliency_map)
                if args.rnn_type == 'static':
                    rnn_states_val = sess.run([s.h for s in rnn_states],
                                              feed_dict={x: batch_x, y: batch_y, x_w: batch_x_w})
                    rnn_states_grad_val = \
                        sess.run(rnn_states_grad, feed_dict={x: batch_x, y: batch_y, x_w: batch_x_w})
                    rnn_vis = RNNVis()
                    rnn_vis.plot_hidden_grad(np.transpose(np.stack(rnn_states_val), [1, 0, 2]),
                                             np.transpose(np.stack(rnn_states_grad_val), [1, 0, 2]),
                                             sequence=np.reshape(batch_x, [batch_size, h, w]), shape=[1, h])
        # summarize model
        #if args.tf_summary and i % args.tf_summary == 0:
        #    logging.info('summarize model to "{}" at epochs {}'.format(args.tf_summary_path, i))
        #    summary = sess.run([merged_summary], feed_dict={x: batch_x, y: batch_y, x_w: batch_x_w})
        #    train_writer.add_summary(summary, i)
        # eval model
        if args.eval and i % args.eval == 0:
            act = input('press "c" to continue')
            while act != 'c':
                batch = next_batch(args.data, 1, h, w,
                                   mean_match_query_term=mean_match_query_term,
                                   mean_match_doc_term=mean_match_doc_term,
                                   dist=distribution)
                model_out_val, loss_val = sess.run([model_out, loss], feed_dict={
                    x: np.expand_dims(batch[0], axis=3),
                    y: np.expand_dims(batch[1], axis=3),
                    x_w: batch[2]})
                print('matrix:')
                with printoptions(precision=3, suppress=True):
                    eval_matrix = np.rint(np.abs(batch[0][0] * w)).astype(int)
                    print(eval_matrix)
                    print('TF: {}'.format(np.sum(eval_matrix)))
                if args.arch == 'cnn':
                    print('target: {0:.3f}, output: {1:.3f}'.format(
                        batch[1][0, h-1, w-1], model_out_val[0, 0]))
                else:
                    print('target: {0:.3f}, output: {1:.3f}'.format(
                        batch[1][0, h-1, w-1], model_out_val[0, h-1, w-1, 0]))
                print('loss: {0:.3f}'.format(loss_val))
                act = input('press "c" to continue')


if __name__ == '__main__':
    main()