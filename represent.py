import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import numpy as np
from cnn import cnn, mlp

def dynamic_split_and_pad_map_fn(x, sp):
  # Always split on the first dimension.
  max_len = tf.reduce_max(sp) # max piece length
  piece_num = tf.shape(sp)[0] # number of pieces
  ind = tf.range(piece_num)
  sp_start = tf.reduce_sum(tf.expand_dims(sp, 0) * \
    tf.cast(tf.expand_dims(ind, 1)>tf.expand_dims(ind, 0), dtype=tf.int32), axis=1)
  def get_padding(p, axis, pad):
      rank = len(p.get_shape())
      padding = [[0,0]] * rank
      padding[axis] = pad
      return padding
  def fn(elems):
    start, offset = elems
    this_piece = x[start:start+offset]
    this_piece = tf.pad(this_piece, get_padding(this_piece, 0, [0, max_len-offset]), 
      'CONSTANT', constant_values=0)
    return this_piece
  piece = tf.map_fn(fn, [sp_start, sp], dtype=x.dtype, 
    parallel_iterations=1000, name='dynamic_split_map')
  return piece

def dynamic_split_and_pad_while_loop(x, sp, axis):
  max_len = tf.reduce_max(sp) # max piece length
  piece_num = tf.shape(sp)[0] # number of pieces
  ind = tf.constant(0)
  piece = tf.TensorArray(dtype=x.dtype, size=piece_num, name='piece', clear_after_read=True)
  def cond(ind, x, sp, piece, max_len):
    return ind < tf.shape(sp)[0]
  def body(ind, x, sp, piece, max_len):
    slice_begin = [0] * len(x.get_shape())
    slice_size = [-1] * len(x.get_shape())
    slice_size[axis] = sp[ind]
    this_piece = tf.slice(x, slice_begin, slice_size)
    def get_padding(p, axis, pad):
      rank = len(p.get_shape())
      padding = [[0,0]] * rank
      padding[axis] = pad
      return padding
    this_piece = tf.pad(this_piece, get_padding(this_piece, axis, [0, max_len-sp[ind]]), 
      'CONSTANT', constant_values=0)
    piece = piece.write(ind, this_piece)
    slice_begin[axis] = sp[ind]
    slice_size[axis] = -1
    x = tf.slice(x, slice_begin, slice_size)
    return ind+1, x, sp, piece, max_len
  _, _, _, piece, _ = tf.while_loop(cond, body,
    [ind, x, sp, piece, max_len],
    parallel_iterations=1,
  )
  return piece.stack()

def whole_rnn(seq, seq_len, emb, **kwargs):
  bs = tf.shape(seq)[0]
  rnn_size = kwargs['rnn_size']
  input_size = emb.get_shape()[1].value
  rnn = tf.contrib.cudnn_rnn.CudnnRNNRelu(num_layers=1, num_units=rnn_size, 
    input_size=input_size, input_mode='linear_input', direction='unidirectional')
  rnn_param = tf.get_variable('rnn_params', shape=[rnn_size+input_size+2, rnn_size])
  initial_state = tf.zeros((1, bs, rnn_size), dtype=tf.float32)
  rnn_input = tf.nn.embedding_lookup(emb, tf.transpose(seq))
  outputs, state = rnn(rnn_input, initial_state, rnn_param)
  # cudnn rnn does not support seq len, we need to do it manually.
  seq_len = tf.expand_dims(seq_len, axis=0)
  ind = tf.expand_dims(tf.range(tf.shape(outputs)[0]), axis=1)
  state = tf.reshape(tf.boolean_mask(outputs, tf.equal(ind, seq_len-1)), [-1, rnn_size])
  #state = tf.Print(state, [state[:10, :5]], message='state:', summarize=50)
  # split state based on piece_num
  state = tf.expand_dims(state, axis=1)
  return state, tf.ones([bs], dtype=tf.int32)

def piece_rnn(cond, cond_value, seq, seq_len, emb, **kwargs):
  # get length of each piece and number of pieces of each sample
  bs = tf.shape(cond)[0]
  max_seq_len = tf.shape(seq)[1]
  cond_pad = tf.pad(cond, [[0,0], [1,1]], 'CONSTANT', constant_values=False)
  cond_start = tf.logical_and(tf.logical_not(cond_pad[:, :-2]), cond)
  cond_end = tf.logical_and(tf.logical_not(cond_pad[:, 2:]), cond)
  ind = tf.expand_dims(tf.range(max_seq_len), axis=0)
  ind = tf.tile(ind, [bs, 1])
  ind_start = tf.boolean_mask(ind, cond_start)
  ind_end = tf.boolean_mask(ind, cond_end)
  piece_len = ind_end - ind_start + 1
  piece_len_dim = tf.shape(piece_len)[0]
  piece_num = tf.reduce_sum(tf.cast(cond_start, dtype=tf.int32), axis=1)
  # select piece
  piece = tf.boolean_mask(seq, cond)
  # word weight
  piece_word_weight = tf.boolean_mask(cond_value, cond)
  # split piece based on piece_len
  piece = dynamic_split_and_pad_map_fn(piece, piece_len)
  piece_word_weight = dynamic_split_and_pad_map_fn(piece_word_weight, piece_len)
  # time major
  piece = tf.transpose(piece, [1, 0])
  piece_word_weight = tf.transpose(piece_word_weight, [1, 0])
  piece = tf.nn.embedding_lookup(emb, piece)
  # word weighting
  piece = piece * tf.expand_dims(piece_word_weight, axis=2)
  # print debug
  piece_len = tf.Print(piece_len, [piece_len], 
    message='piece len:', summarize=50)
  piece_len = tf.Print(piece_len, [tf.reduce_max(piece_len)], 
    message='piece len max:', summarize=1)
  piece_len = tf.Print(piece_len, [tf.reduce_mean(tf.cast(piece_len, dtype=tf.float32))], 
    message='piece len ave:', summarize=1)
  piece_len = tf.Print(piece_len, [piece_len_dim], 
    message='all piece num:', summarize=1)
  #piece_num = tf.Print(piece_num, [piece_num], message='piece_num:', summarize=50)
  # represent each piece
  print('rnn size: {}'.format(kwargs['rnn_size']))
  # traditional RNN
  #rnn_cell = tf.nn.rnn_cell.BasicRNNCell(kwargs['rnn_size'])
  #rnn_cell = tf.nn.rnn_cell.GRUCell(kwargs['rnn_size'])
  #rnn_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(kwargs['rnn_size'])
  #initial_state = rnn_cell.zero_state(piece_len_dim, dtype=tf.float32)
  #outputs, state = tf.nn.dynamic_rnn(rnn_cell, piece, 
  #  initial_state=initial_state, sequence_length=piece_len, dtype=tf.float32, time_major=True, 
  #  swap_memory=False)
  # CuDNN RNN
  rnn_size = kwargs['rnn_size']
  input_size = emb.get_shape()[1].value
  rnn = tf.contrib.cudnn_rnn.CudnnRNNRelu(num_layers=1, num_units=rnn_size, 
    input_size=input_size, input_mode='linear_input', direction='unidirectional')
  rnn_param_size = rnn.params_size()
  '''
  with tf.control_dependencies(None):
    # Variable cannot be initialized from a tensor created in while_loop.
    # This is an ugly workaround.
    rnn_init_param = tf.random_uniform([(rnn_size+input_size+2)*rnn_size], -0.1, 0.1)
  rnn_param = tf.get_variable('rnn_params', initializer=rnn_init_param, validate_shape=False)
  '''
  #rnn_param = tf.get_variable('rnn_params', shape=[(rnn_size+input_size+2)*rnn_size])
  rnn_param = tf.get_variable('rnn_params', shape=[rnn_size+input_size+2, rnn_size])
  initial_state = tf.zeros((1, piece_len_dim, rnn_size), dtype=tf.float32)
  outputs, state = rnn(piece, initial_state, rnn_param)
  # cudnn rnn does not support seq len, we need to do it manually.
  piece_len = tf.expand_dims(piece_len, axis=0)
  ind = tf.expand_dims(tf.range(tf.shape(outputs)[0]), axis=1)
  state = tf.reshape(tf.boolean_mask(outputs, tf.equal(ind, piece_len-1)), [-1, rnn_size])
  state = tf.Print(state, [state[:10, :5]], message='state:', summarize=50)
  # split state based on piece_num
  state = dynamic_split_and_pad_map_fn(state, piece_num)
  return state, piece_num

def cnn_rnn(match_matrix, dq_size, query, query_emb, doc, doc_emb, word_vector, **kwargs):
  '''
  Use CNN to select regions and RNN to represent
  '''
  bs = tf.shape(match_matrix)[0]
  max_q_len = tf.shape(query)[1]
  max_d_len = tf.shape(doc)[1]
  thres = kwargs['threshold']
  time = kwargs['time']
  state_ta = kwargs['state_ta']
  print('threshold: {}'.format(thres))
  # use CNN to choose regions
  with vs.variable_scope('CNNRegionFinder'):
    cnn_decision_value = cnn(tf.expand_dims(match_matrix, axis=-1), 
      architecture=[(5, 5, 1, 4), (1, 1), (1, 1, 4, 1), (1, 1)], activation='tanh')
    cnn_decision_value = tf.Print(cnn_decision_value, [cnn_decision_value[0, :20, :5, 0]], 
      message='cnn', summarize=100)
    cnn_decision_value = tf.reshape(cnn_decision_value, tf.shape(cnn_decision_value)[:3])
    # mask out the words beyond the boundary
    doc_mask = tf.expand_dims(tf.range(max_d_len), dim=0) < tf.reshape(dq_size[:1], [bs, 1])
    query_mask = tf.expand_dims(tf.range(max_q_len), dim=0) < tf.reshape(dq_size[1:], [bs, 1])
    mask = tf.cast(tf.logical_and(tf.expand_dims(doc_mask, axis=2), 
      tf.expand_dims(query_mask, axis=1)), dtype=tf.float32)
    cnn_decision_value = (cnn_decision_value-thres) * mask + thres
    # make decision by "or"
    doc_decision_value = tf.reduce_max(cnn_decision_value, axis=2)
    query_decision_value = tf.reduce_max(cnn_decision_value, axis=1)
    doc_decision = doc_decision_value > thres
    query_decision = query_decision_value > thres
    # print debug
    doc_decision = tf.Print(doc_decision, 
      [tf.reduce_mean(tf.reduce_sum(tf.cast(doc_decision, tf.int32), axis=1))], 
      message='avg all doc piece:')
    query_decision = tf.Print(query_decision, 
      [tf.reduce_mean(tf.reduce_sum(tf.cast(query_decision, tf.int32), axis=1))], 
      message='avg all query piece:')
  '''
  with vs.variable_scope('ThresholdRegionFinder'):
    # randomly mask some part to avoid very long sequence
    ber = tf.contrib.distributions.Bernoulli(probs=0.99)
    ber = ber.sample([1, tf.shape(match_matrix)[1], 1])
    match_matrix *= tf.cast(ber, dtype=match_matrix.dtype)
    decision = tf.logical_or(match_matrix>0.4, match_matrix<-0.0)
    doc_decision = tf.reduce_any(decision, axis=2)
    doc_decision_value = tf.ones_like(doc_decision, dtype=tf.float32)
    #query_decision = tf.reduce_any(decision, axis=1)
    query_decision = tf.reduce_any(tf.logical_not(tf.equal(match_matrix, 0.0)), axis=1) # special for query
    query_decision_value = tf.ones_like(query_decision, dtype=tf.float32)
    doc_decision = tf.Print(doc_decision, 
      [tf.reduce_mean(tf.reduce_sum(tf.cast(doc_decision, tf.float32), axis=1))], 
      message='avg all doc piece:')
    query_decision = tf.Print(query_decision, 
      [tf.reduce_mean(tf.reduce_sum(tf.cast(query_decision, tf.float32), axis=1))], 
      message='avg all query piece:')
  '''
  with vs.variable_scope('RNNRegionRepresenter'):
    doc_piece_emb, doc_piece_num = piece_rnn(
      doc_decision, doc_decision_value, doc, dq_size[0], word_vector, **kwargs)
    #doc_piece_emb, doc_piece_num = whole_rnn(doc, dq_size[0], word_vector, **kwargs)
    vs.get_variable_scope().reuse_variables()
    query_piece_emb, query_piece_num = piece_rnn(
      query_decision, query_decision_value, query, dq_size[1], word_vector, **kwargs)
    #query_piece_emb, query_piece_num = whole_rnn(query, dq_size[1], word_vector, **kwargs)
  with vs.variable_scope('KNRMAggregator'):
    # interaction
    match_matrix = tf.matmul(doc_piece_emb, tf.transpose(query_piece_emb, [0, 2, 1]))
    max_q_len = tf.shape(query_piece_emb)[1]
    max_d_len = tf.shape(doc_piece_emb)[1]
    dq_size = tf.stack([doc_piece_num, query_piece_num], axis=0)
    # use K-NRM
    if 'input_mu' in kwargs and kwargs['input_mu'] != None:
        input_mu = kwargs['input_mu']
    else:
        input_mu = np.array(list(range(-10,10+1,2)))/10
    #input_mu = [1.0]
    number_of_bin = len(input_mu)-1
    input_sigma =  [0.1] * number_of_bin + [0.1]
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
    #representation = tf.log(1+representation) # log is used in K-NRM
    # use a MLP to model interactions between evidence of different strength
    #mlp_arch = [number_of_bin+1, number_of_bin+1]
    #print('use MLP with structure {}'.format(mlp_arch))
    #representation = mlp(representation, architecture=mlp_arch, activation='relu')
    return state_ta, representation