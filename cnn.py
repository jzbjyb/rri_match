import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np
from utils import printoptions


class DynamicMaxPooling(object):
    def __init__(self, dim=2, shape=None):
        self.conv_dim = dim
        self.shape = shape


    def __call__(self, x, dpool_index, pool_size, strides, padding, name=None):
        x_expand = tf.gather_nd(x, dpool_index)
        if self.conv_dim == 1:
            size = [self.shape[0] / pool_size[0]]
            if size[0] != 1:
                x_pool = tf.layers.max_pooling1d(x_expand, pool_size=size, strides=size, padding=padding, name=name)
            else:
                x_pool = x_expand
        elif self.conv_dim == 2:
            size = [self.shape[0] / pool_size[0], self.shape[1] / pool_size[1]]
            if size[0] != 1 or size[1] != 1:
                x_pool = tf.layers.max_pooling2d(x_expand, pool_size=size, strides=size, padding=padding, name=name)
            else:
                x_pool = x_expand
        return x_pool


    @staticmethod
    def dynamic_pooling_index_1d(leng, max_len, compress_ratio=1):
        bs = tf.shape(leng)[0]
        dpool_bias = 0
        if max_len % compress_ratio != 0:
            dpool_bias = 1
        cur_max_len = max_len // compress_ratio + dpool_bias
        leng = leng // compress_ratio
        bidx = tf.ones([bs, cur_max_len], dtype=tf.int32) * tf.expand_dims(tf.range(bs), axis=-1)
        leng = tf.maximum(1, leng)
        stride = tf.cast(cur_max_len / leng, dtype=tf.float32)
        lidx = tf.cast(tf.cast(tf.expand_dims(tf.range(cur_max_len), axis=0), dtype=tf.float32) / 
            tf.expand_dims(stride, axis=-1), dtype=tf.int32)
        return tf.stack([bidx, lidx], axis=-1)


    @staticmethod
    def dynamic_pooling_index_1d_np(leng, max_len, compress_ratio=1):
        def dpool_index_(batch_idx, leng, max_len):
            if leng == 0:
                stride = max_len
            else:
                stride = 1.0 * max_len / leng
            idx = [int(i / stride) for i in range(max_len)]
            index = np.stack([np.ones_like(idx) * batch_idx, idx], axis=-1)
            return index
        index = []
        dpool_bias = 0
        if max_len % compress_ratio != 0:
            dpool_bias = 1
        cur_max_len = max_len // compress_ratio + dpool_bias
        for i in range(len(leng)):
            index.append(dpool_index_(i, leng[i] // compress_ratio, cur_max_len))
        return np.array(index)


    @staticmethod
    def dynamic_pooling_index_2d_np(len1, len2, max_len1, max_len2, compress_ratio1=1, compress_ratio2=1):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            if len1_one == 0:
                stride1 = max_len1
            else:
                stride1 = 1.0 * max_len1 / len1_one

            if len2_one == 0:
                stride2 = max_len2
            else:
                stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i / stride1) for i in range(max_len1)]
            idx2_one = [int(i / stride2) for i in range(max_len2)]
            mesh2, mesh1 = np.meshgrid(idx2_one, idx1_one)
            index_one = np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2], axis=-1)
            return index_one
        index = []
        dpool_bias1 = dpool_bias2 = 0
        if max_len1 % compress_ratio1 != 0:
            dpool_bias1 = 1
        if max_len2 % compress_ratio2 != 0:
            dpool_bias2 = 1
        cur_max_len1 = max_len1 // compress_ratio1 + dpool_bias1
        cur_max_len2 = max_len2 // compress_ratio2 + dpool_bias2
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i] // compress_ratio1, 
                         len2[i] // compress_ratio2, cur_max_len1, cur_max_len2))
        return np.array(index)


def cnn(x, architecture=[(3, 3, 1, 16), (1, 2, 2, 1)], activation='relu', dpool_index=None):
    if activation not in {None, 'relu', 'tanh'}:
        raise Exception('not supported activation')
    if len(architecture) <= 0 or len(architecture) % 2 != 0:
        raise Exception('cnn architecture bug')
    if len(architecture[0]) == 3:
        conv_dim = 1
    elif len(architecture[0]) == 4:
        conv_dim = 2
    else:
        raise Exception('architecture not correct')
    if conv_dim == 1:
        shape = x.get_shape().as_list()[1:2]
    elif conv_dim == 2:
        shape = x.get_shape().as_list()[1:3]
    last = x
    for i in range(len(architecture) // 2):
        layer = i + 1
        conv_size = architecture[i*2]
        pool_size = architecture[i*2+1]
        with tf.name_scope('conv{}'.format(layer)) as scope:
            #kernel = tf.Variable(tf.truncated_normal(conv_size, dtype=tf.float32, stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights', shape=conv_size, dtype=tf.float32, initializer=\
                                     tf.truncated_normal_initializer(mean=0.0, stddev=1e-1, dtype=tf.float32))
            tf.add_to_collection('conv_kernel', kernel)
            if conv_dim == 1:
                conv = tf.nn.conv1d(last, kernel, 1, padding='SAME')
            elif conv_dim == 2:
                conv = tf.nn.conv2d(last, kernel, [1, 1, 1, 1], padding='SAME')
            #biases = tf.Variable(tf.constant(0.0, shape=[conv_size[-1]], dtype=tf.float32),
            #                     trainable=True, name='biases')
            biases = tf.get_variable('biases', shape=[conv_size[-1]], dtype=tf.float32, initializer=\
                                     tf.zeros_initializer())
            out = tf.nn.bias_add(conv, biases)
            if activation == 'relu':
                out = tf.nn.relu(out, name=scope)
            elif activation == 'tanh':
                out = tf.nn.tanh(out, name=scope)
            tf.add_to_collection('feature_map', out)
        with tf.name_scope('pool{}'.format(layer)) as scope:
            if dpool_index is not None and layer == 1: # dynamic pooling
                dynamic_max_pool = DynamicMaxPooling(dim=conv_dim, shape=shape)
                out = dynamic_max_pool(out, dpool_index, pool_size=pool_size, strides=pool_size, 
                                       padding='SAME', name='pool')
            else: # conventional pooling
                if conv_dim == 1 and pool_size[0] != 1:
                    out = tf.layers.max_pooling1d(out, pool_size=pool_size, strides=pool_size, 
                                                  padding='SAME', name='pool')
                elif conv_dim == 2 and (pool_size[0] != 1 or pool_size[1] != 1):
                    out = tf.layers.max_pooling2d(out, pool_size=pool_size, strides=pool_size, 
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