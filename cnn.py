import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import matplotlib.pyplot as plt
import math
import numpy as np
from utils import printoptions


def mlp(x, architecture=[10], activation='relu'):
  with vs.variable_scope('MultipleLayerPerceptron'):
    for i, layer in enumerate(architecture):
      with vs.variable_scope('Layer{}'.format(i+1)):
        hidden_size = architecture[i]
        w = tf.get_variable('weight', shape=[x.get_shape()[1], hidden_size])
        b = tf.get_variable('bias', shape=[hidden_size], 
          initializer=tf.constant_initializer())
        x = tf.nn.bias_add(tf.matmul(x, w), b)
        if activation == 'relu':
          x = tf.nn.relu(x)
        elif activation == 'tanh':
          x = tf.nn.tanh(x)
  return x


def knrm(match_matrix, max_q_len, max_d_len, dq_size, input_mu, input_sigma, match_matrix_mask=None,
  use_log=True, use_mlp=False, sum_per_query=False):
  bs = tf.shape(match_matrix)[0]
  number_of_bin = len(input_mu) - 1
  mu = tf.constant(input_mu, dtype=tf.float32)
  sigma = tf.constant(input_sigma, dtype=tf.float32)
  mu = tf.reshape(mu, [1, 1, 1, number_of_bin + 1])
  sigma = tf.reshape(sigma, [1, 1, 1, number_of_bin + 1])
  # kernelize
  match_matrix = tf.expand_dims(match_matrix, axis=-1)
  match_matrix = tf.exp(-tf.square(match_matrix - mu) / (tf.square(sigma) * 2))
  # have to use mask because the weight is masked
  query_mask = tf.expand_dims(tf.range(max_q_len), dim=0) < tf.reshape(dq_size[1:], [bs, 1])
  doc_mask = tf.expand_dims(tf.range(max_d_len), dim=0) < tf.reshape(dq_size[:1], [bs, 1])
  query_mask = tf.cast(tf.reshape(query_mask, [bs, 1, max_q_len, 1]), dtype=tf.float32)
  doc_mask = tf.cast(tf.reshape(doc_mask, [bs, max_d_len, 1, 1]), dtype=tf.float32)
  match_matrix = match_matrix * query_mask * doc_mask
  if match_matrix_mask is not None:
    # mask using other matrix
    match_matrix *= tf.expand_dims(match_matrix_mask, axis=-1)
  # sum
  if sum_per_query:
    # first, sum over document terms
    representation = tf.reduce_sum(match_matrix, axis=1)
  else:
    representation = tf.reduce_sum(match_matrix, axis=[1, 2])
  if use_log:
    # log is used in K-NRM
    representation = tf.log(1 + representation)
  if sum_per_query:
    # second, sum over query terms
    representation = tf.reduce_sum(representation, axis=1) / 100 # scaling
  if use_mlp:
    # use a MLP to model interactions between evidence of different strength
    mlp_arch = [number_of_bin+1, number_of_bin+1]
    print('use MLP with structure {}'.format(mlp_arch))
    representation = mlp(representation, architecture=mlp_arch, activation='relu')
  return representation


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
  def dynamic_pooling_index_2d(len1, len2, max_len1, max_len2, compress_ratio1=1, compress_ratio=1):
    # TODO: don't support compress_ratio
    batch_size = tf.shape(len1)[0]
    # convert zeros to ones
    len1 = len1 + (len1 == 0)
    len2 = len2 + (len2 == 0)
    # compute stride
    stride1 = max_len1 / len1
    stride2 = max_len2 / len2
    # compute index
    idx1 = tf.cast(tf.expand_dims(tf.range(max_len1), dim=0), dtype=tf.float64) / tf.expand_dims(stride1, dim=1)
    idx2 = tf.cast(tf.expand_dims(tf.range(max_len2), dim=0), dtype=tf.float64) / tf.expand_dims(stride2, dim=1)
    idx1 = tf.cast(idx1, dtype=tf.int32)
    idx2 = tf.cast(idx2, dtype=tf.int32)
    # mesh
    mesh1 = tf.tile(tf.expand_dims(idx1, 2), [1, 1, max_len2])
    mesh2 = tf.tile(tf.expand_dims(idx2, 1), [1, max_len1, 1])
    # construct index
    bidx = tf.ones_like(mesh1) * tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
    return tf.stack([bidx, mesh1, mesh2], axis=3)

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
  if activation not in {None, 'relu', 'tanh', 'sigmoid'}:
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
    with vs.variable_scope('conv{}'.format(layer)):
      #kernel = tf.Variable(tf.truncated_normal(conv_size, dtype=tf.float32, stddev=1e-1), name='weights')
      kernel = tf.get_variable('weights', shape=conv_size, dtype=tf.float32, initializer=None)
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
        out = tf.nn.relu(out)
      elif activation == 'tanh':
        out = tf.nn.tanh(out)
      elif activation == 'sigmoid':
        out = tf.nn.sigmoid(out)
      tf.add_to_collection('feature_map', out)
    with vs.variable_scope('pool{}'.format(layer)):
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
    if len(image.shape) == 3:
      image = np.expand_dims(image, axis=-1)
      saliency_map = np.expand_dims(saliency_map, axis=-1)
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