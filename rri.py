import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs
from cnn import cnn, DynamicMaxPooling, mlp
from represent import cnn_rnn, cnn_text_rnn
jumper = tf.load_op_library('./jumper.so')
DELTA = 1e-5 # used to avoid dividing zero

def batch_slice(batch, start, offset, pad_values=None):
  bs = tf.shape(batch)[0]
  max_offset = tf.reduce_max(offset)
  min_last = tf.reduce_min(tf.shape(batch)[1] - start)
  pad_len = tf.reduce_max([max_offset - min_last, 0])
  rank = len(batch.get_shape())
  remain = tf.shape(batch)[2:]
  # padding
  batch_pad = tf.pad(batch, [[0, 0], [0, pad_len]] + [[0, 0] for r in range(rank - 2)], 'CONSTANT',
             constant_values=pad_values)
  dim_len = tf.shape(batch_pad)[1]
  # gather
  ind_center = start + tf.range(bs) * dim_len
  ind_region = tf.reshape(tf.expand_dims(ind_center, axis=-1) + tf.expand_dims(tf.range(max_offset), axis=0), [-1])
  region = tf.reshape(tf.gather(tf.reshape(batch_pad, tf.concat([[-1], remain], axis=0)), ind_region),
            tf.concat([[bs, max_offset], remain], axis=0))
  return region


def batch_slice_slow(batch, start, offset, pad_values=None):
  bs = tf.shape(batch)[0]
  max_offset = tf.reduce_max(offset)
  min_last = tf.reduce_min(tf.shape(batch)[1] - start)
  pad_len = tf.reduce_max([max_offset - min_last, 0])
  rank = len(batch.get_shape())
  remain = tf.shape(batch)[2:]
  # padding
  batch_pad = tf.pad(batch, [[0, 0], [0, pad_len]] + [[0, 0] for r in range(rank - 2)], 'CONSTANT',
             constant_values=pad_values)
  dim_len = tf.shape(batch_pad)[1]
  # gather
  start = tf.expand_dims(start, axis=-1)
  ind = tf.expand_dims(tf.range(dim_len), axis=0)
  ind = tf.logical_and(ind >= start, ind < start + max_offset)
  ind = tf.cast(ind, dtype=tf.int32)
  _, region = tf.dynamic_partition(batch_pad, ind, 2)
  region = tf.reshape(region, tf.concat([[bs, max_offset], remain], axis=0))
  return region


def batch_where(cond, xs, ys):
  if xs == 1:
    xs = tf.ones_like(ys[0])
    xs = [xs] * len(ys)
  return [tf.where(cond, xs[i], ys[i]) for i in range(len(xs))]


def get_glimpse_location(match_matrix, dq_size, location, glimpse):
  '''
  get next glimpse location (g_t+1) based on last jump location (j_t)
  '''
  if glimpse == 'fix_hard':
    gp_d_position = tf.cast(tf.floor(location[:, 0] + location[:, 2]), dtype=tf.int32)
    gp_d_offset = tf.reduce_min([tf.ones_like(dq_size[:, 0], dtype=tf.int32) * glimpse_fix_size,
                   dq_size[:, 0] - gp_d_position], axis=0)
    glimpse_location = tf.stack([tf.cast(gp_d_position, dtype=tf.float32),
                   tf.zeros_like(location[:, 1]),
                   tf.cast(gp_d_offset, dtype=tf.float32),
                   tf.cast(dq_size[:, 1], dtype=tf.float32)], axis=1)
  elif glimpse == 'all_next_hard':
    gp_d_position = location[0] + location[2]
    gp_d_offset = dq_size[0] - gp_d_position
    glimpse_location = [gp_d_position, tf.zeros_like(location[1]), gp_d_offset, dq_size[1]]
  else:
    raise NotImplementedError()
  return glimpse_location


def get_jump_location(match_matrix, dq_size, location, jump, **kwargs):
  '''
  get next jump location (j_t+1) based on glimpse location (g_t+1)
  '''
  if jump == 'max_hard':
    max_d_offset = tf.cast(tf.floor(tf.reduce_max(location[:, 2])), dtype=tf.int32)
    # padding
    match_matrix_pad = tf.pad(match_matrix, [[0, 0], [0, max_d_offset], [0, 0]], 'CONSTANT',
                  constant_values=sys.float_info.min)
    d_len = tf.shape(match_matrix_pad)[1]
    start = tf.cast(tf.floor(location[:, 0]), dtype=tf.int32)
    gp_ind_center = start + tf.range(bs) * d_len
    gp_ind_region = tf.reshape(tf.expand_dims(gp_ind_center, axis=-1) +
                   tf.expand_dims(tf.range(max_d_offset), axis=0), [-1])
    glimpse_region = tf.reshape(tf.gather(tf.reshape(match_matrix_pad, [-1, max_q_len]), gp_ind_region),
                  [-1, max_d_offset, max_q_len])
    d_loc = tf.argmax(tf.reduce_max(tf.abs(glimpse_region), axis=2), axis=1) + start
    new_location = tf.stack([tf.cast(d_loc, dtype=tf.float32),
                 location[:, 1], tf.ones([bs]), location[:, 3]], axis=1)
  elif jump == 'min_density_hard':
    #new_location = jumper.min_density(match_matrix=match_matrix, dq_size=dq_size, 
    #    location=tf.stack(location, axis=0), min_density=min_density)
    # there is no need to use multi-thread op, because this is fast and thus not the bottleneck
    new_location = jumper.min_density_multi_cpu(
      match_matrix=match_matrix, dq_size=dq_size, 
      location=tf.stack(location, axis=0), min_density=kwargs['min_density'],
      min_jump_offset=kwargs['min_jump_offset'], use_ratio=False, only_one=False)
    new_location = [new_location[i] for i in range(4)]
  elif jump == 'all_hard' or jump == 'all_soft':
    new_location = location
  else:
    raise NotImplementedError()
  new_location = [tf.stop_gradient(nl) for nl in new_location]
  return new_location


def get_representation(match_matrix, dq_size, query, query_emb, doc, doc_emb, word_vector, location, \
             represent, **kwargs):
  '''
  get the representation based on location (j_t+1)
  '''
  bs = tf.shape(query)[0]
  max_q_len = tf.shape(query)[1]
  max_d_len = tf.shape(doc)[1]
  word_vector_dim = word_vector.get_shape().as_list()[1]
  separate = kwargs['separate']
  state_ta = kwargs['state_ta']
  location_ta = kwargs['location_ta']
  doc_repr_ta = kwargs['doc_repr_ta']
  query_repr_ta = kwargs['query_repr_ta']
  time = kwargs['time']
  is_stop = kwargs['is_stop']
  min_density = kwargs['min_density']
  query_weight = kwargs['query_weight']
  doc_weight = kwargs['doc_weight']
  cur_location = location_ta.read(time)
  cur_next_location = location_ta.read(time + 1)
  with vs.variable_scope('ReprCond'):
    pass
    # use last representation if the location remains unchanged
    #doc_reuse = \
    #    tf.logical_and(tf.reduce_all(tf.equal(cur_location[:, 0:4:2], cur_next_location[:, 0:4:2])), 
    #                   tf.greater_equal(time, 1))
    #query_reuse = \
    #    tf.logical_and(tf.reduce_all(tf.equal(cur_location[:, 1:4:2], cur_next_location[:, 1:4:2])), 
    #                   tf.greater_equal(time, 1))
  if represent == 'sum_hard':
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, lambda: state_ta.write(0, tf.zeros([bs, 1])))
    start = tf.cast(tf.floor(location[:, :2]), dtype=tf.int32)
    end = tf.cast(tf.floor(location[:, :2] + location[:, 2:]), dtype=tf.int32)
    ind = tf.constant(0)
    representation_ta = tf.TensorArray(dtype=tf.float32, size=bs,
                       name='representation_ta', clear_after_read=False)
    def body(i, m, s, e, r):
      r_i = tf.reduce_sum(m[i][s[i, 0]:e[i, 0], s[i, 1]:e[i, 1]])
      r = r.write(i, tf.reshape(r_i, [1]))
      return i + 1, m, s, e, r
    _, _, _, _, representation_ta = \
      tf.while_loop(lambda i, m, s, e, r: i < bs, body,
              [ind, match_matrix, start, end, representation_ta],
              parallel_iterations=1000)
    representation = representation_ta.stack()
  elif represent == 'sum_match_matrix_hard':
    '''
    Directly sum all the elements in match_matrix without considering location.
    The elements in the padding part of the matrix should be zero in this case. 
    '''
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, 
      lambda: state_ta.write(0, tf.zeros([bs, 1], dtype=tf.float32)))
    representation = tf.reduce_sum(match_matrix, axis=[1,2])
    representation = tf.expand_dims(representation, axis=1)
  elif represent == 'cnn_rnn_hard':
    state_ta, representation = cnn_rnn(match_matrix, dq_size, query, query_emb, doc, doc_emb, 
      word_vector, threshold=0.0, **kwargs)
  elif represent == 'cnn_text_rnn_hard':
    use_cudnn = True
    if kwargs['direction'] == 'bidirectional':
      use_cudnn = False
    state_ta, representation = cnn_text_rnn(match_matrix, dq_size, query, query_emb, doc, doc_emb, 
      word_vector, use_combine=True, query_as_unigram=True, threshold=0.4, use_single=False,
      use_cudnn=use_cudnn, activation='tanh', **kwargs)
  elif represent == 'sum_match_matrix_kernel_hard':
    '''
    K-NRM-like kernels
    '''
    if 'input_mu' in kwargs and kwargs['input_mu'] != None:
      input_mu = kwargs['input_mu']
    else:
      input_mu = np.array(list(range(-10,10+1,2)))/10
    #input_mu = [1.0]
    number_of_bin = len(input_mu)-1
    input_sigma =  [0.1] * number_of_bin + [0.00001]
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, 
      lambda: state_ta.write(0, tf.zeros([bs, number_of_bin+1], dtype=tf.float32)))
    print('mu: {}, sigma: {}'.format(input_mu, input_sigma))
    mu = tf.constant(input_mu, dtype=tf.float32)
    sigma = tf.constant(input_sigma, dtype=tf.float32)
    mu = tf.reshape(mu, [1, 1, 1, number_of_bin+1])
    sigma = tf.reshape(sigma, [1, 1, 1, number_of_bin+1])
    # kernelize
    match_matrix = tf.expand_dims(match_matrix, axis=-1)
    # totally discard some part of the matrix
    #print('discard some part of the matrix in K-NRM')
    #discard_match_matrix = tf.logical_and(match_matrix>=-0.1, match_matrix<=0.5)
    match_matrix = tf.exp(-tf.square(match_matrix-mu)/(tf.square(sigma)*2))
    # have to use mask because the weight is masked
    query_mask = tf.expand_dims(tf.range(max_q_len), dim=0) < tf.reshape(dq_size[1:], [bs, 1])
    doc_mask = tf.expand_dims(tf.range(max_d_len), dim=0) < tf.reshape(dq_size[:1], [bs, 1])
    query_mask = tf.cast(tf.reshape(query_mask, [bs, 1, max_q_len, 1]), dtype=tf.float32)
    doc_mask = tf.cast(tf.reshape(doc_mask, [bs, max_d_len, 1, 1]), dtype=tf.float32)
    match_matrix = match_matrix * query_mask * doc_mask
    # totally discard some part of the matrix
    #match_matrix *= 1-tf.cast(discard_match_matrix, dtype=tf.float32)
    # sum and log
    representation = tf.reduce_sum(match_matrix, axis=[1, 2])
    # this is for manually masking out some kernels
    #                 [-1 -0.8 -0.6 -0.4 -0.2  0.   0.2  0.4  0.6  0.8  1.]
    #representation *= [[0,  0,    0,   1,   1,  0,    0,   0,   1,   1,  1]]
    representation = tf.log(1+representation) # log is used in K-NRM
    # use a MLP to model interactions between evidence of different strength
    #mlp_arch = [number_of_bin+1, number_of_bin+1]
    #print('use MLP with structure {}'.format(mlp_arch))
    #representation = mlp(representation, architecture=mlp_arch, activation='relu')
  elif represent == 'sum_match_matrix_topk_weight_thre_kernel_hard':
    '''
    Use different kernels to first mask the match_matrix into several matrices. 
    Then apply weight to each matrix to get several scores. The exact match is theoretically
    equivalent to tfidf.
    '''
    #input_mu = np.array(list(range(number_of_bin+1))) * (1/number_of_bin)
    input_mu = [0.8, 0.9, 1.0]
    number_of_bin = len(input_mu)-1
    input_sigma =  [0.1] * number_of_bin + [0.00001]
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, 
      lambda: state_ta.write(0, tf.zeros([bs, number_of_bin+1], dtype=tf.float32)))
    print('mu: {}, sigma: {}'.format(input_mu, input_sigma))
    mu = tf.constant(input_mu, dtype=tf.float32)
    sigma = tf.constant(input_sigma, dtype=tf.float32)
    mu = tf.reshape(mu, [1, 1, 1, number_of_bin+1])
    sigma = tf.reshape(sigma, [1, 1, 1, number_of_bin+1])
    # don't have to use mask because the weight is masked
    match_matrix_max = tf.reduce_max(match_matrix, axis=2, keep_dims=True)
    match_matrix = match_matrix * tf.cast(tf.equal(match_matrix, match_matrix_max), 
      dtype=tf.float32)
    match_matrix = tf.expand_dims(match_matrix, axis=-1)
    match_matrix = tf.exp(-tf.square(match_matrix-mu) / \
      (tf.square(sigma)*2))
    match_matrix = match_matrix * tf.reshape(doc_weight, [bs, max_d_len, 1, 1]) * \
      tf.reshape(query_weight, [bs, 1, max_q_len, 1])
    representation = tf.reduce_sum(match_matrix, axis=[1, 2])
    representation = tf.log(1+representation) # log is used in K-NRM
  elif represent == 'sum_match_matrix_topk_weight_thres_hard':
    '''
    Directly sum topk elements in each row of match_matrix without considering location.
    In other words, each document choose topk similar words (>= thres) from the query.
    The elements in the padding part of the matrix should be zero in this case.
    '''
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, 
      lambda: state_ta.write(0, tf.zeros([bs, 1], dtype=tf.float32)))
    match_matrix_max = tf.reduce_max(match_matrix, axis=2, keep_dims=True)
    mmm_cond = tf.logical_and(tf.equal(match_matrix, match_matrix_max), match_matrix>=0.5)
    match_matrix = match_matrix * tf.cast(mmm_cond, dtype=tf.float32)
    match_matrix = match_matrix * tf.expand_dims(doc_weight, axis=2) * \
      tf.expand_dims(query_weight, axis=1)
    representation = tf.reduce_sum(match_matrix, axis=[1, 2])
    representation = tf.expand_dims(representation, axis=1)
  elif represent == 'sum_match_matrix_topk_weight_hard':
    '''
    Directly sum topk elements in each row of match_matrix without considering location.
    In other words, each document choose topk similar words from the query.
    The elements in the padding part of the matrix should be zero in this case.
    '''
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, 
      lambda: state_ta.write(0, tf.zeros([bs, 1], dtype=tf.float32)))
    #representation = tf.reduce_sum(tf.nn.top_k(match_matrix, k=1).values, axis=[1,2])
    match_matrix_max = tf.reduce_max(match_matrix, axis=2, keep_dims=True)
    match_matrix = match_matrix * tf.cast(tf.equal(match_matrix, match_matrix_max), 
      dtype=tf.float32)
    match_matrix = match_matrix * tf.expand_dims(doc_weight, axis=2) * \
      tf.expand_dims(query_weight, axis=1)
    representation = tf.reduce_sum(match_matrix, axis=[1, 2])
    representation = tf.expand_dims(representation, axis=1)
  elif represent == 'log_tf_hard':
    '''
    Use log(tf + 1) to calculate the term frequency for each term in the query and sum them.
    This is mimicing ltc.lnc vector space model, but not use document length normalization 
    and df to make fair comparison with neural models.
    '''
    # initialize the first element of state_ta
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, 
      lambda: state_ta.write(0, tf.zeros([bs, 1], dtype=tf.float32)))
    d_start, d_offset, q_offset = location[0], location[2], location[3]
    local_match_matrix = batch_slice(match_matrix, d_start, d_offset, pad_values=0)
    local_match_matrix = tf.maximum(local_match_matrix, 0)
    term_freq = tf.reduce_sum(local_match_matrix, axis=1)
    term_freq = tf.log(term_freq + 1)
    representation = tf.reduce_sum(term_freq, axis=1, keep_dims=True)
    representation = representation / \
      tf.expand_dims(tf.cast(q_offset, dtype=term_freq.dtype)+DELTA, axis=1)
  elif represent == 'mean_pooling':
    '''
    Calculate mean similarity between document words and each query word. This is for
    comparison with log_tf_hard, which only uses exact match.
    '''
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, 
      lambda: state_ta.write(0, tf.zeros([bs, 1], dtype=tf.float32)))
    d_start, d_offset = location[0], location[2]
    local_match_matrix = batch_slice(match_matrix, d_start, d_offset, pad_values=0)
    local_match_matrix = tf.maximum(local_match_matrix, 0)
    term_freq = tf.reduce_sum(local_match_matrix, axis=1)
    term_freq = term_freq / tf.expand_dims(tf.cast(d_offset, dtype=tf.float32) + DELTA, axis=1)
    representation = tf.reduce_sum(term_freq, axis=1, keep_dims=True)
  elif represent == 'interaction_copy_hard':
    '''
    This represent method just copy the match_matrix selected by current region to state_ta.
    Must guarantee that the offset of doc is the same for different step/jump. Offset on query
    is not important because we select regions only based on location of doc.
    Otherwise, the TensorArray will raise inconsistent shape exception.
    '''
    d_start, d_offset = location[0], location[2]
    local_match_matrix = batch_slice(match_matrix, d_start, d_offset, pad_values=0)
    # initialize the first element of state_ta
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, 
      lambda: state_ta.write(0, tf.zeros_like(local_match_matrix)))
    representation = local_match_matrix
  elif represent == 'interaction_cnn_mask_hard':
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, lambda: state_ta.write(0, tf.zeros([bs, 200])))
    # in this implementation, we mask the match_matrix uisng min_density threshold and
    # apply CNN on it. Note that this implementation does not depend on the location, which
    # makes the tf.while useless.
    if 'max_jump_offset' not in kwargs or 'max_jump_offset2' not in kwargs:
      raise ValueError('max_jump_offset and max_jump_offset2 must be set when InterCNN is used')
    max_jump_offset = kwargs['max_jump_offset']
    max_jump_offset2 = kwargs['max_jump_offset2']
    d_size, q_size = dq_size[0], dq_size[1]
    local_match_matrix = match_matrix * tf.cast(
      tf.reduce_max(match_matrix, axis=2, keep_dims=True) >= \
      tf.reshape(min_density, [bs, 1, 1]), dtype=match_matrix.dtype)
    local_match_matrix = tf.pad(local_match_matrix, 
      [[0, 0], [0, max_jump_offset-tf.shape(local_match_matrix)[1]], 
      [0, max_jump_offset2-tf.shape(local_match_matrix)[2]]], 'CONSTANT', constant_values=0)
    local_match_matrix.set_shape([None, max_jump_offset, max_jump_offset2])
    local_match_matrix = tf.expand_dims(local_match_matrix, 3)
    with vs.variable_scope('MaksInterCNN'):
      inter_dpool_index = DynamicMaxPooling.dynamic_pooling_index_2d(d_size, q_size, 
        max_jump_offset, max_jump_offset2)
      inter_repr = cnn(local_match_matrix, architecture=[(5, 5, 1, 8), (5, 5)], activation='relu',
        dpool_index=inter_dpool_index)
      representation = tf.reshape(inter_repr, [bs, -1])
  elif represent == 'interaction_cnn_resize_hard':
    state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, lambda: state_ta.write(0, tf.zeros([bs, 200])))
    # in this implementation, we don't calculate similarity matrix again
    if 'max_jump_offset' not in kwargs or 'max_jump_offset2' not in kwargs:
      raise ValueError('max_jump_offset and max_jump_offset2 must be set when InterCNN is used')
    max_jump_offset = kwargs['max_jump_offset']
    max_jump_offset2 = kwargs['max_jump_offset2']
    start = tf.cast(tf.floor(location[:, :2]), dtype=tf.int32)
    offset = tf.cast(tf.floor(location[:, 2:]), dtype=tf.int32)
    d_start, d_offset = start[:, 0], offset[:, 0]
    q_start, q_offset = start[:, 1], offset[:, 1]
    d_end = d_start + d_offset - 1
    q_end = q_start + q_offset - 1
    d_start = d_start / dq_size[:, 0]
    d_end = d_end / dq_size[:, 0]
    q_start = q_start / dq_size[:, 1]
    q_end = q_end / dq_size[:, 1]
    local_match_matrix = tf.image.crop_and_resize(
      tf.expand_dims(match_matrix, -1),
      boxes=tf.cast(tf.stack([d_start, q_start, d_end, q_end], axis=-1), dtype=tf.float32),
      box_ind=tf.range(bs),
      crop_size=[max_jump_offset, max_jump_offset2],
      method='bilinear',
      name='local_interaction'
    )
    with vs.variable_scope('InterCNN'):
      inter_repr = cnn(local_match_matrix, 
        architecture=[(5, 5, 1, 8), (max_jump_offset/5, max_jump_offset2/5)], 
        activation='relu',
        dpool_index=None)
      representation = tf.reshape(inter_repr, [bs, -1])
  elif represent in {'rnn_hard', 'cnn_hard', 'interaction_cnn_hard'}:
    if represent in {'rnn_hard', 'cnn_hard'}:
      state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, lambda: state_ta.write(0, tf.zeros([bs, 1])))
    elif represent in {'interaction_cnn_hard'}:
      state_ta = tf.cond(tf.greater(time, 0), lambda: state_ta, lambda: state_ta.write(0, tf.zeros([bs, 200])))
    d_start, q_start, d_offset, q_offset = location
    d_region = batch_slice(doc, d_start, d_offset, pad_values=0)
    q_region = batch_slice(query, q_start, q_offset, pad_values=0)
    d_region = tf.nn.embedding_lookup(word_vector, d_region)
    q_region = tf.nn.embedding_lookup(word_vector, q_region)
    if represent == 'interaction_cnn_hard':
      # This implementation seems to be slow, wo don't use it
      if 'max_jump_offset' not in kwargs or 'max_jump_offset2' not in kwargs:
        raise ValueError('max_jump_offset and max_jump_offset2 must be set when InterCNN is used')
      max_jump_offset = kwargs['max_jump_offset']
      max_jump_offset2 = kwargs['max_jump_offset2']
      local_match_matrix = tf.matmul(d_region, tf.transpose(q_region, [0, 2, 1]))
      local_match_matrix = tf.pad(local_match_matrix, 
        [[0, 0], [0, max_jump_offset-tf.shape(local_match_matrix)[1]], 
        [0, max_jump_offset2-tf.shape(local_match_matrix)[2]]], 'CONSTANT', constant_values=0)
      local_match_matrix.set_shape([None, max_jump_offset, max_jump_offset2])
      local_match_matrix = tf.expand_dims(local_match_matrix, 3)
      with vs.variable_scope('InterCNN'):
        inter_dpool_index = DynamicMaxPooling.dynamic_pooling_index_2d(d_offset, q_offset, 
          max_jump_offset, max_jump_offset2)
        inter_repr = cnn(local_match_matrix, architecture=[(5, 5, 1, 8), (5, 5)], activation='relu',
        #inter_repr = cnn(local_match_matrix, architecture=[(5, 5, 1, 16), (500, 10), (5, 5, 16, 16), (1, 1), (5, 5, 16, 16), (10, 1), (5, 5, 16, 100), (25, 10)], activation='relu',
          dpool_index=inter_dpool_index)
        representation = tf.reshape(inter_repr, [bs, -1])
    elif represent == 'rnn_hard':
      #rnn_cell = tf.nn.rnn_cell.BasicRNNCell(kwargs['rnn_size'])
      rnn_cell = tf.nn.rnn_cell.GRUCell(kwargs['rnn_size'])
      initial_state = rnn_cell.zero_state(bs, dtype=tf.float32)
      d_outputs, d_state = tf.nn.dynamic_rnn(rnn_cell, d_region, initial_state=initial_state,
                           sequence_length=d_offset, dtype=tf.float32)
      q_outputs, q_state = tf.nn.dynamic_rnn(rnn_cell, q_region, initial_state=initial_state,
                           sequence_length=q_offset, dtype=tf.float32)
      representation = tf.reduce_sum(d_state * q_state, axis=1, keep_dims=True)
    elif represent == 'cnn_hard':
      if 'max_jump_offset' not in kwargs:
        raise ValueError('max_jump_offset must be set when CNN is used')
      max_jump_offset = kwargs['max_jump_offset']
      doc_after_pool_size = max_jump_offset
      doc_arch = [[3, word_vector_dim, 4], [doc_after_pool_size]]
      query_arch = [[3, word_vector_dim, 4], [max_jump_offset]]
      #doc_arch, query_arch = [[3, word_vector_dim, 4], [10]], [[3, word_vector_dim, 4], [5]]
      doc_repr_ta = tf.cond(tf.greater(time, 0), lambda: doc_repr_ta, 
                  lambda: doc_repr_ta.write(0, tf.zeros([bs, 10, doc_arch[-2][-1]])))
      query_repr_ta = tf.cond(tf.greater(time, 0), lambda: query_repr_ta, 
                  lambda: query_repr_ta.write(0, tf.zeros([bs, 5, query_arch[-2][-1]])))
      def get_doc_repr():
        nonlocal d_region, max_jump_offset, word_vector_dim, separate, d_offset, doc_arch, doc_after_pool_size
        d_region = tf.pad(d_region, [[0, 0], [0, max_jump_offset - tf.shape(d_region)[1]], [0, 0]], 
                  'CONSTANT', constant_values=0)
        d_region.set_shape([None, max_jump_offset, word_vector_dim])
        with vs.variable_scope('DocCNN' if separate else 'CNN'):
          doc_dpool_index = DynamicMaxPooling.dynamic_pooling_index_1d(d_offset, max_jump_offset)
          doc_repr = cnn(d_region, architecture=doc_arch, activation='relu',
                   dpool_index=doc_dpool_index)
        with vs.variable_scope('LengthOrderAwareMaskPooling'):
          mask_prob = tf.minimum(tf.ceil(doc_after_pool_size ** 2 / dq_size[:, 0]), doc_after_pool_size) / 50
          # length-aware mask
          mask_ber = tf.distributions.Bernoulli(probs=mask_prob)
          mask = tf.transpose(mask_ber.sample([doc_after_pool_size]), [1, 0])
          # order-aware pooling
          #mask_for_zero = tf.cast(tf.expand_dims(tf.range(doc_after_pool_size), axis=0) < \
          #    (doc_after_pool_size - tf.reduce_sum(mask, axis=1, keep_dims=True)), dtype=tf.int32)
          #mask = tf.cast(tf.concat([mask, mask_for_zero], axis=1), dtype=tf.bool)
          #doc_repr = tf.boolean_mask(tf.concat([doc_repr, tf.zeros_like(doc_repr)], axis=1), mask)
          #doc_repr = tf.reshape(doc_repr, [bs, doc_after_pool_size, doc_arch[-2][-1]])
          # normal pooling
          doc_repr = doc_repr * tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)
          # pooling
          doc_repr = tf.layers.max_pooling1d(doc_repr, pool_size=[5], strides=[5],
                             padding='SAME', name='pool')
        return doc_repr
      def get_query_repr():
        nonlocal q_region, max_jump_offset, word_vector_dim, separate, q_offset, query_arch
        q_region = tf.pad(q_region, [[0, 0], [0, max_jump_offset - tf.shape(q_region)[1]], [0, 0]],
                  'CONSTANT', constant_values=0)
        q_region.set_shape([None, max_jump_offset, word_vector_dim])
        with vs.variable_scope('QueryCNN' if separate else 'CNN'):
          if not separate:
            vs.get_variable_scope().reuse_variables()
          query_dpool_index = DynamicMaxPooling.dynamic_pooling_index_1d(q_offset, max_jump_offset)
          query_repr = cnn(q_region, architecture=query_arch, activation='relu', 
                   dpool_index=query_dpool_index)
          query_repr = tf.layers.max_pooling1d(query_repr, pool_size=[10], strides=[10],
                             padding='SAME', name='pool')
        return query_repr
      doc_repr = tf.cond(doc_reuse, lambda: doc_repr_ta.read(time), get_doc_repr)
      query_repr = tf.cond(query_reuse, lambda: query_repr_ta.read(time), get_query_repr)
      #doc_repr = tf.cond(tf.constant(False), lambda: doc_repr_ta.read(time), get_doc_repr)
      #query_repr = tf.cond(tf.constant(False), lambda: query_repr_ta.read(time), get_query_repr)
      doc_repr_ta = doc_repr_ta.write(time + 1, tf.where(is_stop, doc_repr_ta.read(time), doc_repr))
      query_repr_ta = query_repr_ta.write(time + 1, tf.where(is_stop, query_repr_ta.read(time), query_repr))
      cnn_final_dim = 10 * doc_arch[-2][-1] + 5 * query_arch[-2][-1]
      #cnn_final_dim = doc_arch[-1][0] * doc_arch[-2][-1] + query_arch[-1][0] * query_arch[-2][-1]
      dq_repr = tf.reshape(tf.concat([doc_repr, query_repr], axis=1), [-1, cnn_final_dim])
      dq_repr = tf.nn.dropout(dq_repr, kwargs['keep_prob'])
      dq_repr = tf.layers.dense(inputs=dq_repr, units=4, activation=tf.nn.relu)
      representation = tf.layers.dense(inputs=dq_repr, units=1, activation=tf.nn.relu)
  elif represent == 'test':
    representation = tf.ones_like(location[:, :1])
  else:
    raise NotImplementedError()
  state_ta = state_ta.write(time + 1, representation)
  #state_ta = state_ta.write(time + 1, tf.where(is_stop, state_ta.read(time), representation))
  return state_ta, doc_repr_ta, query_repr_ta


def rri(query, doc, dq_size, max_jump_step, word_vector, interaction=['dot'], glimpse='fix_hard', glimpse_fix_size=None,
    min_density=None, use_ratio=False, min_jump_offset=1, jump='max_hard', represent='sum_hard', separate=False,
    all_position=True, direction='unidirectional', aggregate='max', rnn_size=None, max_jump_offset=None,
    max_jump_offset2=None, keep_prob=1.0, query_weight=None, doc_weight=None, input_mu=None):
  bs = tf.shape(query)[0]
  max_q_len = tf.shape(query)[1]
  max_d_len = tf.shape(doc)[1]
  word_vector_dim = word_vector.get_shape().as_list()[1]
  '''
  Because this implementation involves a while_loop, lots of strided_slice and cast 
  will lower the performance and GPU utility. So dq_size and location is transposed to 
  (dim, batch_size) to accelerate while_loop.
  '''
  dq_size_t = tf.transpose(dq_size)
  with vs.variable_scope('Embed'):
    query_emb = tf.nn.embedding_lookup(word_vector, query)
    doc_emb = tf.nn.embedding_lookup(word_vector, doc)
  with vs.variable_scope('Match'):
    # The first string in interaction signifies how to interact query embeding 
    # and document embedding. The following elements signifies how to update match_matrix
    # obtained by the first step.
    if type(interaction) != list:
      interaction = [interaction]
    # match_matrix is of shape (batch_size, max_d_len, max_q_len)
    if interaction[0] == 'indicator':
      match_matrix = tf.cast(tf.equal(tf.expand_dims(doc, axis=2), tf.expand_dims(query, axis=1)),
                   dtype=tf.float32)
      query_boundary = tf.expand_dims(tf.range(max_q_len), dim=0) < dq_size[:, 1:]
      doc_boundary = tf.expand_dims(tf.range(max_d_len), dim=0) < dq_size[:, :1]
      match_matrix = match_matrix * \
        tf.cast(tf.logical_and(tf.expand_dims(query_boundary, axis=1), 
        tf.expand_dims(doc_boundary, axis=2)), dtype=tf.float32)
    else:
      if interaction[0] == 'dot':
        match_matrix = tf.matmul(doc_emb, tf.transpose(query_emb, [0, 2, 1]))
      elif interaction[0] == 'cosine':
        doc_emb_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_emb), axis=2, keep_dims=True))
        doc_emb_norm += tf.cast(tf.equal(doc_emb_norm, 0), dtype=tf.float32)
        query_emb_norm = tf.sqrt(tf.reduce_sum(tf.square(query_emb), axis=2, keep_dims=True))
        query_emb_norm += tf.cast(tf.equal(query_emb_norm, 0), dtype=tf.float32)
        doc_emb = doc_emb / doc_emb_norm
        query_emb = query_emb / query_emb_norm
        match_matrix = tf.matmul(doc_emb, tf.transpose(query_emb, [0, 2, 1]))
    for interaction_update in interaction[1:]:
      if interaction_update == 'weight':
        if query_weight is None or doc_weight is None:
          raise Exception('no weight is provided')
        match_matrix = match_matrix * tf.expand_dims(doc_weight, axis=2) * \
          tf.expand_dims(query_weight, axis=1)
    if min_density != None:
      if use_ratio:
        '''
        # using max min value is not guaranteed to find the threshold
        with vs.variable_scope('RatioDensity'):
          density = tf.reduce_max(match_matrix, 2)
          mean_density = tf.reduce_mean(density, 1)
          max_density = tf.reduce_max(density, 1)
          min_density = (max_density - mean_density) * min_density + mean_density
        '''
        # only kep top min_density percentage words with maximum density in the document
        with vs.variable_scope('PercentileDensity'):
          print('use percentile density')
          density = tf.reduce_max(match_matrix, 2)
          density, _ = tf.nn.top_k(density, 
            tf.cast(tf.cast(max_d_len, dtype=tf.float32)*min_density, dtype=tf.int32))
          top_k = tf.cast(tf.ceil(tf.cast(dq_size_t[0], dtype=tf.float32)*min_density), 
            dtype=tf.int32) - 1
          min_density = tf.squeeze(batch_slice(density, top_k, 
            tf.ones_like(top_k), pad_values=0))
      else:
        print('use global density')
        min_density = tf.ones_like(dq_size[:, 0], dtype=tf.float32) * min_density
  with vs.variable_scope('SelectiveJump'):
    if jump.endswith('hard'):
      location_ta_dtype = tf.int32
    elif jump.endswith('soft'):
      location_ta_dtype = tf.float32
    location_ta = tf.TensorArray(dtype=location_ta_dtype, size=1, name='location_ta',
                   clear_after_read=False, dynamic_size=True) # (d_ind,q_ind,d_len,q_len)
    location_ta = location_ta.write(0, tf.zeros([4, bs], dtype=location_ta_dtype)) # start from the top-left corner
    state_ta = tf.TensorArray(dtype=tf.float32, size=1, name='state_ta', clear_after_read=False, 
                  dynamic_size=True)
    query_repr_ta = tf.TensorArray(dtype=tf.float32, size=1, name='query_repr_ta', clear_after_read=False, 
                     dynamic_size=True)
    doc_repr_ta = tf.TensorArray(dtype=tf.float32, size=1, name='doc_repr_ta', clear_after_read=False, 
                   dynamic_size=True)
    step = tf.zeros([bs], dtype=tf.int32)
    total_offset = tf.zeros([bs], dtype=location_ta_dtype)
    is_stop = tf.zeros([bs], dtype=tf.bool)
    time = tf.constant(0)
    def cond(time, is_stop, step, state_ta, doc_repr_ta, query_repr_ta, location_ta, dq_size_t, total_offset):
      return tf.logical_and(tf.logical_not(tf.reduce_all(is_stop)),
                  tf.less(time, tf.constant(max_jump_step)))
    def body(time, is_stop, step, state_ta, doc_repr_ta, query_repr_ta, location_ta, dq_size_t, total_offset):
      cur_location = location_ta.read(time)
      # The basic optimization is that:
      # (1) each step ("glimpse," "jump," and "represent") use the location as if 
      #     it is in correct dtype (int32 or float32). dtype conversion is conducted 
      #     when needed to reduce the number of calls to cast.
      # (2) location is divided into four parts (d_start, q_start, d_offset, q_offset)
      #     to reduce the number of calls to stride_slice.
      cur_location_l = [cur_location[i] for i in range(4)]
      #time = tf.Print(time, [time], message='time:')
      with vs.variable_scope('Glimpse'):
        if glimpse.endswith('soft') and jump.endswith('hard'):
          cur_location_in_glimpse = [tf.cast(cl, dtype=tf.float32) for cl in cur_location_l]
        elif glimpse.endswith('hard') and jump.endswith('soft'):
          cur_location_in_glimpse = [tf.cast(cl, dtype=tf.int32) for cl in cur_location_l]
        else:
          cur_location_in_glimpse = cur_location_l
        glimpse_location = get_glimpse_location(
          match_matrix, dq_size_t, cur_location_in_glimpse, glimpse)
        # stop when the start index overflow
        if glimpse.endswith('soft'):
          new_stop = tf.reduce_any(tf.stack(glimpse_location[:2], axis=0) > \
            tf.cast(dq_size_t-1, tf.float32), axis=0)
        elif glimpse.endswith('hard'):
          new_stop = tf.reduce_any(tf.stack(glimpse_location[:2], axis=0) > \
            dq_size_t-1, axis=0)
        glimpse_location = batch_where(new_stop, cur_location_in_glimpse, glimpse_location)
        is_stop = tf.logical_or(is_stop, new_stop)
      with vs.variable_scope('Jump'):
        if glimpse.endswith('soft') and jump.endswith('hard'):
          glimpse_location = [tf.cast(gl, dtype=tf.int32) for gl in glimpse_location]
        elif glimpse.endswith('hard') and jump.endswith('soft'):
          glimpse_location = [tf.cast(gl, dtype=tf.float32) for gl in glimpse_location]
        new_location = get_jump_location(match_matrix, dq_size_t, glimpse_location, jump, 
          min_density=min_density, min_jump_offset=min_jump_offset)
        if max_jump_offset != None:
          # truncate long document offset
          new_location = [new_location[0], new_location[1], 
            tf.minimum(new_location[2], max_jump_offset), new_location[3]]
        if max_jump_offset2 != None:
          # truncate long query offset
          new_location = [new_location[0], new_location[1], new_location[2],
            tf.minimum(new_location[3], max_jump_offset2)]
        # stop when the start index overflow
        if jump.endswith('soft'):
          new_stop = tf.reduce_any(tf.stack(new_location[:2], axis=0) > \
            tf.cast(dq_size_t-1, tf.float32), axis=0)
        elif jump.endswith('hard'):
          new_stop = tf.reduce_any(tf.stack(new_location[:2], axis=0) > \
            dq_size_t-1, axis=0)
        is_stop = tf.logical_or(is_stop, new_stop)
        # location_ta and total_offset alwasy have the same dtype with new_location 
        # which is generated by jump.
        location_ta = location_ta.write(time + 1, 
          tf.stack(batch_where(is_stop, cur_location_l, new_location), axis=0))
        # total length considered
        total_offset += tf.where(is_stop, tf.zeros_like(total_offset), new_location[2])
      with vs.variable_scope('Represent'):
        cur_next_location = location_ta.read(time + 1)
        cur_next_location_l = [cur_next_location[i] for i in range(4)]
        # location_one_out is to prevent duplicate time-consuming calculation
        #location_one_out = batch_where(is_stop, 1, cur_next_location_l)
        location_one_out = cur_next_location_l
        if represent.endswith('hard') and jump.endswith('soft'):
          location_one_out = [tf.cast(loo, dtype=tf.int32) for loo in location_one_out]
        elif represent.endswith('soft') and jump.endswith('hard'):
          location_one_out =[tf.cast(loo, dtype=tf.float32) for loo in location_one_out]
        state_ta, doc_repr_ta, query_repr_ta = \
          get_representation(match_matrix, dq_size_t, query, query_emb, doc, doc_emb, word_vector, \
                     location_one_out, represent, max_jump_offset=max_jump_offset, \
                     max_jump_offset2=max_jump_offset2, rnn_size=rnn_size, keep_prob=keep_prob, \
                     separate=separate, location_ta=location_ta, state_ta=state_ta, doc_repr_ta=doc_repr_ta, \
                     query_repr_ta=query_repr_ta, time=time, is_stop=is_stop, min_density=min_density, \
                     query_weight=query_weight, doc_weight=doc_weight, input_mu=input_mu,
                     all_position=all_position, direction=direction)
      step = step + tf.where(is_stop, tf.zeros([bs], dtype=tf.int32), tf.ones([bs], dtype=tf.int32))
      # early stop if all is processed
      if jump.endswith('soft'):
        new_stop = tf.reduce_any(
          tf.stack(new_location[:2], axis=0)+tf.stack(new_location[2:], axis=0) > \
          tf.cast(dq_size_t-1, tf.float32), axis=0)
      elif jump.endswith('hard'):
        new_stop = tf.reduce_any(
          tf.stack(new_location[:2], axis=0)+tf.stack(new_location[2:], axis=0) > \
          dq_size_t-1, axis=0)
      is_stop = tf.logical_or(is_stop, new_stop)
      return time + 1, is_stop, step, state_ta, doc_repr_ta, query_repr_ta, location_ta, dq_size_t, total_offset
    _, is_stop, step, state_ta, doc_repr_ta, query_repr_ta, location_ta, dq_size_t, total_offset = \
      tf.while_loop(cond, body, [time, is_stop, step, state_ta, doc_repr_ta, query_repr_ta, 
              location_ta, dq_size_t, total_offset], parallel_iterations=1)
  with vs.variable_scope('Aggregate'):
    states = state_ta.stack()
    location = location_ta.stack()
    location = tf.cast(location, dtype=tf.float32)
    location = tf.transpose(location, [2, 0, 1])
    stop_ratio = tf.reduce_mean(tf.cast(is_stop, tf.float32))
    complete_ratio = tf.reduce_mean(tf.reduce_min(
      [(location[:, -1, 0] + location[:, -1, 2]) / tf.cast(dq_size[:, 0], dtype=tf.float32),
       tf.ones([bs], dtype=tf.float32)], axis=0))
    if aggregate == 'max':
      signal = tf.reduce_max(states, 0)
    elif aggregate == 'sum':
      signal = tf.reduce_sum(states, 0) - states[-1] * \
        tf.cast(tf.expand_dims(time - step, axis=-1), dtype=tf.float32)
    elif aggregate == 'interaction_concat':
      '''
      Concatenate all the state (local match matrix) in state_ta (without the first element 
      because it is initialized as zeros). Then apply CNN.
      '''
      infered_max_d_len = max_jump_step * max_jump_offset
      infered_max_q_len = max_jump_offset2
      concat_match_matrix = tf.reshape(tf.transpose(states[1:], 
        tf.concat([[1, 0], tf.range(len(states.get_shape()))[2:]], axis=0)), 
        [bs, -1, max_q_len])
      concat_match_matrix = tf.pad(concat_match_matrix, 
        [[0, 0], [0, infered_max_d_len-tf.shape(concat_match_matrix)[1]], 
        [0, infered_max_q_len-tf.shape(concat_match_matrix)[2]]], 
        'CONSTANT', constant_values=0)
      concat_match_matrix.set_shape([None, infered_max_d_len, infered_max_q_len])
      concat_match_matrix = tf.expand_dims(concat_match_matrix, 3)
      with vs.variable_scope('ConcateCNN'):
        concat_dpool_index = DynamicMaxPooling.dynamic_pooling_index_2d(
          tf.cast(total_offset, tf.int32), dq_size[:, 1], 
          infered_max_d_len, infered_max_q_len)
        concat_repr = cnn(concat_match_matrix, architecture=[(5, 5, 1, 8), (5, 5)], 
          activation='relu', dpool_index=concat_dpool_index)
        signal = tf.reshape(concat_repr, [bs, 200])
    return signal, {'step': step, 'location': location, 'match_matrix': match_matrix, 
            'complete_ratio': complete_ratio, 'is_stop': is_stop, 'stop_ratio': stop_ratio,
            'doc_emb': doc_emb, 'total_offset': total_offset, 'signal': signal, 
            'states': states, 'min_density': min_density}