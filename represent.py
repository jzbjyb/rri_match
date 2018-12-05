import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import numpy as np
from cnn import cnn, mlp

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

def dynamic_split_and_pad(x, sp, axis, max_len):
  sp_dim = tf.shape(sp)[0]
  sp = tf.pad(sp, [[0, max_len-sp_dim]])
  sp.set_shape([max_len])
  print('start split')
  piece = tf.split(x, sp, axis=axis)
  print('end split')
  max_piece_len = tf.reduce_max([tf.shape(p)[axis] for p in piece])
  
  print('start piece')
  piece = [tf.cond(tf.greater(tf.shape(p)[axis], 0), 
    lambda: tf.pad(p, get_padding(p, axis, [0, max_piece_len-tf.shape(p)[axis]]), 
    'CONSTANT', constant_values=0),  p) for p in piece]
  print('end piece')
  def get_shape(x, axis, new_shape):
    shape = tf.shape(axis)
    shape = [shape[s] for s in range(len(x.get_shape()))]
    shape = shape[:axis] + new_shape + shape[axis+1:]
    return shape
  return tf.reshape(tf.concat(piece, axis=axis), get_shape(x, axis, [sp_dim, max_piece_len]))

def piece_rnn(cond, seq, seq_len, emb, **kwargs):
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
  # split piece based on piece_len
  piece = tf.boolean_mask(seq, cond)
  piece = dynamic_split_and_pad_while_loop(piece, piece_len, axis=0)  
  # represent each piece
  rnn_cell = tf.nn.rnn_cell.GRUCell(kwargs['rnn_size'])
  initial_state = rnn_cell.zero_state(piece_len_dim, dtype=tf.float32)
  outputs, state = tf.nn.dynamic_rnn(rnn_cell, tf.nn.embedding_lookup(emb, piece), 
    initial_state=initial_state, sequence_length=piece_len, dtype=tf.float32)
  # split state based on piece_num
  state = dynamic_split_and_pad_while_loop(state, piece_num, axis=0)
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
    cnn_decision = cnn(tf.expand_dims(match_matrix, axis=-1), 
      architecture=[(5, 5, 1, 4), (1, 1), (5, 5, 4, 1), (1, 1)], activation='relu')
    cnn_decision = tf.reshape(cnn_decision, tf.shape(cnn_decision)[:3]) >= thres
    doc_decision = tf.reduce_any(cnn_decision, axis=2)
    query_decision = tf.reduce_any(cnn_decision, axis=1)
    # mask out the words beyond the boundary
    doc_mask = tf.expand_dims(tf.range(max_d_len), dim=0) < tf.reshape(dq_size[:1], [bs, 1])
    query_mask = tf.expand_dims(tf.range(max_q_len), dim=0) < tf.reshape(dq_size[1:], [bs, 1])
    doc_decision = tf.logical_and(doc_decision, doc_mask)
    query_decision = tf.logical_and(query_decision, query_mask)
    doc_decision = tf.Print(doc_decision, 
      [tf.reduce_mean(tf.reduce_sum(tf.cast(doc_decision, tf.int32), axis=1))], message='doc:')
    query_decision = tf.Print(query_decision, 
      [tf.reduce_mean(tf.reduce_sum(tf.cast(query_decision, tf.int32), axis=1))], message='query:')
  with vs.variable_scope('RNNRegionRepresenter'):
    # TODO: how to deal with max_piece_len_dim and max_piece_num_dim
    doc_piece_emb, doc_piece_num = piece_rnn(
      doc_decision, doc, dq_size[0], word_vector, **kwargs)
    vs.get_variable_scope().reuse_variables()
    query_piece_emb, query_piece_num = piece_rnn(
      query_decision, query, dq_size[1], word_vector, **kwargs)
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
        input_mu = np.array(list(range(-10,10+1,2)))/50
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