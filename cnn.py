import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np
from utils import printoptions


def cnn(input, architecture=[(1, 3, 3, 16), (1, 2, 2, 1)], activation='relu'):
    if activation not in {None, 'relu', 'tanh'}:
        raise Exception('not supported activation')
    if len(architecture) <= 0 or len(architecture) % 2 != 0:
        raise Exception('cnn architecture bug')
    last = input
    for i in range(len(architecture) // 2):
        layer = i + 1
        conv_size = architecture[i*2]
        pool_size = architecture[i*2+1]
        with tf.name_scope('conv{}'.format(layer)) as scope:
            kernel = tf.Variable(tf.truncated_normal(conv_size, dtype=tf.float32, stddev=1e-1),
                                 name='weights')
            tf.add_to_collection('conv_kernel', kernel)
            conv = tf.nn.conv2d(last, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[conv_size[-1]], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            if activation == 'relu':
                out = tf.nn.relu(out, name=scope)
            elif activation == 'tanh':
                out = tf.nn.tanh(out, name=scope)
            tf.add_to_collection('feature_map', out)
        with tf.name_scope('pool{}'.format(layer)) as scope:
            out = tf.nn.max_pool(out, ksize=pool_size, strides=pool_size,
                                 padding='SAME', name='pool')
        last = out
    return last


class CNNVis(object):
    SALIENCY_MAX_ROW = 9
    SALIENCY_MAX_COL = 3

    def __init__(self):
        pass


    def set_max_min(self, vmax, vmin):
        self.vmax = vmax
        self.vmin = vmin


    def plot_conv_kernel(self, kernel, name):
        in_c = kernel.shape[2]
        out_c = kernel.shape[3]
        print('vis kernel: {} with size: {}'.format(name, kernel.shape))
        fig, axes = plt.subplots(out_c, in_c, figsize=(10, 7))
        axes = axes.reshape([out_c, in_c])
        for o in range(out_c):
            for i in range(in_c):
                ax = axes[o, i]
                print(o, i)
                print(kernel[:, :, i, o])
                ax.imshow(kernel[:, :, i, o], vmin=self.vmin, vmax=self.vmax,
                          interpolation='none', cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
        try:
            plt.show()
        except KeyboardInterrupt:
            print('plot show interrupted')
            pass


    def plot_saliency_map(self, image, saliency_map):
        bs, h, w, c = saliency_map.shape
        print('vis saliency map with size: {}'.format(saliency_map.shape))
        saliency_map = np.max(np.abs(saliency_map), axis=3)
        smax = np.max(saliency_map)
        imax = np.max(image)
        imin = np.min(image)
        print('saliency max: {}'.format(smax))
        fig, axes = plt.subplots(CNNVis.SALIENCY_MAX_ROW, 2 * CNNVis.SALIENCY_MAX_COL, figsize=(10, 7))
        axes = axes.reshape([CNNVis.SALIENCY_MAX_ROW, 2 * CNNVis.SALIENCY_MAX_COL])
        for b in range(math.ceil(bs / (CNNVis.SALIENCY_MAX_ROW * CNNVis.SALIENCY_MAX_COL))):
            for i in range(CNNVis.SALIENCY_MAX_ROW * CNNVis.SALIENCY_MAX_COL):
                ni = i + b * (CNNVis.SALIENCY_MAX_ROW * CNNVis.SALIENCY_MAX_COL)
                if ni >= bs:
                    break
                r = i // CNNVis.SALIENCY_MAX_COL
                c_img = (i % CNNVis.SALIENCY_MAX_COL) * 2
                c_s = (i % CNNVis.SALIENCY_MAX_COL) * 2 + 1
                ax_img = axes[r, c_img]
                ax_s = axes[r, c_s]
                if c == 1:
                    ax_img.imshow(image[ni, :, :, 0], vmin=imin, vmax=imax,
                              interpolation='none', cmap='gray')
                elif c == 3:
                    ax_img.imshow(image[ni, :, :, :], vmin=imin, vmax=imax,
                                  interpolation='none', cmap='gray')
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                ax_s.imshow(saliency_map[ni, :, :], vmin=0, vmax=np.max(saliency_map[ni, :, :]),
                            interpolation='none', cmap='gray')
                ax_s.set_xticks([])
                ax_s.set_yticks([])
                #with printoptions(precision=2, suppress=True):
                    #print(np.rint(np.abs(image[ni, :, :, 0] * w)).astype(int))
                    #print(saliency_map[i, :, :])
            try:
                plt.show()
            except KeyboardInterrupt:
                print('plot show interrupted')
                pass
            cont = input('press "c" to continue')
            if cont != 'c':
                break