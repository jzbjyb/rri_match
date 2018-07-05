import sys
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from cnn import cnn, DynamicMaxPooling
jumper = tf.load_op_library('./jumper.so')


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
        gp_d_position = tf.cast(tf.floor(location[:, 0] + location[:, 2]), dtype=tf.int32)
        gp_d_offset = dq_size[:, 0] - gp_d_position
        glimpse_location = tf.stack([tf.cast(gp_d_position, dtype=tf.float32),
                                     tf.zeros_like(location[:, 1]),
                                     tf.cast(gp_d_offset, dtype=tf.float32),
                                     tf.cast(dq_size[:, 1], dtype=tf.float32)], axis=1)
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
        #new_location = jumper.min_density(match_matrix=match_matrix, dq_size=dq_size, location=location,
        #                                  min_density=min_density)
        # there is no need to use multi-thread op, because this is fast and thus not the bottleneck
        new_location = jumper.min_density_multi_cpu(
            match_matrix=match_matrix, dq_size=dq_size, location=location, min_density=kwargs['min_density'])
        new_location = tf.stop_gradient(new_location)
    elif jump == 'all':
        new_location = tf.stop_gradient(location)
    elif jump == 'test':
        new_location = location[:, 0] + tf.reduce_min([tf.ones_like(location[:, 1]), location[:, 1]])
        new_location = tf.stack([new_location, location[:, 1], tf.ones([bs]), location[:, 3]], axis=1)
    else:
        raise NotImplementedError()
    return new_location


def get_representation(match_matrix, dq_size, query, query_emb, doc, doc_emb, word_vector, location, \
                       represent, **kwargs):
    '''
    get the representation based on location (j_t+1)
    '''
    bs = tf.shape(query)[0]
    word_vector_dim = word_vector.get_shape().as_list()[1]
    if represent == 'sum_hard':
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
    elif represent == 'rnn_hard_memory_hungry':
        start = tf.cast(tf.floor(location[:, :2]), dtype=tf.int32)
        offset = tf.cast(tf.floor(location[:, 2:]), dtype=tf.int32)
        d_start, d_offset = start[:, 0], offset[:, 0]
        q_start, q_offset = start[:, 1], offset[:, 1]
        d_region = batch_slice(doc_emb, d_start, d_offset, pad_values=0)
        q_region = batch_slice(query_emb, q_start, q_offset, pad_values=0)
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(kwargs['rnn_size'])
        initial_state = rnn_cell.zero_state(bs, dtype=tf.float32)
        d_outputs, d_state = tf.nn.dynamic_rnn(rnn_cell, d_region, initial_state=initial_state,
                                               sequence_length=d_offset, dtype=tf.float32)
        q_outputs, q_state = tf.nn.dynamic_rnn(rnn_cell, q_region, initial_state=initial_state,
                                               sequence_length=q_offset, dtype=tf.float32)
        representation = tf.reduce_sum(d_state * q_state, axis=1, keep_dims=True)
    elif represent in {'rnn_hard', 'cnn_hard'}:
        start = tf.cast(tf.floor(location[:, :2]), dtype=tf.int32)
        offset = tf.cast(tf.floor(location[:, 2:]), dtype=tf.int32)
        d_start, d_offset = start[:, 0], offset[:, 0]
        q_start, q_offset = start[:, 1], offset[:, 1]
        d_region = batch_slice(doc, d_start, d_offset, pad_values=0)
        q_region = batch_slice(query, q_start, q_offset, pad_values=0)
        d_region = tf.nn.embedding_lookup(word_vector, d_region)
        q_region = tf.nn.embedding_lookup(word_vector, q_region)
        if represent == 'rnn_hard':
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
            d_region = tf.pad(d_region, [[0, 0], [0, max_jump_offset - tf.shape(d_region)[1]], [0, 0]], 
                              'CONSTANT', constant_values=0)
            d_region.set_shape([None, max_jump_offset, word_vector_dim])
            q_region = tf.pad(q_region, [[0, 0], [0, max_jump_offset - tf.shape(q_region)[1]], [0, 0]],
                              'CONSTANT', constant_values=0)
            q_region.set_shape([None, max_jump_offset, word_vector_dim])
            doc_arch, query_arch = [[3, word_vector_dim, 4], [10]], [[3, word_vector_dim, 4], [5]]
            cnn_final_dim = doc_arch[-1][0] * doc_arch[-2][-1] + query_arch[-1][0] * query_arch[-2][-1]
            with vs.variable_scope('CNN'):
                doc_dpool_index = DynamicMaxPooling.dynamic_pooling_index_1d(d_offset, max_jump_offset)
                #doc_dpool_index = kwargs['doc_dpool_index']
                doc_repr = cnn(d_region, architecture=doc_arch, activation='relu',
                               dpool_index=doc_dpool_index)
                vs.get_variable_scope().reuse_variables()
                query_dpool_index = DynamicMaxPooling.dynamic_pooling_index_1d(q_offset, max_jump_offset)
                #query_dpool_index = kwargs['query_dpool_index']
                query_repr = cnn(q_region, architecture=query_arch, activation='relu', 
                                 dpool_index=query_dpool_index)
            dq_repr = tf.reshape(tf.concat([doc_repr, query_repr], axis=1), [-1, cnn_final_dim])
            dq_repr = tf.nn.dropout(dq_repr, kwargs['keep_prob'])
            dq_repr = tf.layers.dense(inputs=dq_repr, units=4, activation=tf.nn.relu)
            representation = tf.layers.dense(inputs=dq_repr, units=1, activation=tf.nn.relu)
    elif represent == 'test':
        representation = tf.ones_like(location[:, :1])
    else:
        raise NotImplementedError()
    return representation


def rri(query, doc, dq_size, max_jump_step, word_vector, interaction='dot', glimpse='fix_hard', glimpse_fix_size=None,
        min_density=None, jump='max_hard', represent='sum_hard', aggregate='max', rnn_size=None, max_jump_offset=None,
        query_dpool_index=None, doc_dpool_index=None, keep_prob=1.0):
    bs = tf.shape(query)[0]
    max_q_len = tf.shape(query)[1]
    max_d_len = tf.shape(doc)[1]
    word_vector_dim = word_vector.get_shape().as_list()[1]
    with vs.variable_scope('Embed'):
        query_emb = tf.nn.embedding_lookup(word_vector, query)
        doc_emb = tf.nn.embedding_lookup(word_vector, doc)
    with vs.variable_scope('Match'):
        # match_matrix is of shape (batch_size, max_d_len, max_q_len)
        if interaction == 'indicator':
            match_matrix = tf.cast(tf.equal(tf.expand_dims(doc, axis=2), tf.expand_dims(query, axis=1)),
                                   dtype=tf.float32)
        else:
            if interaction == 'dot':
                match_matrix = tf.matmul(doc_emb, tf.transpose(query_emb, [0, 2, 1]))
            elif interaction == 'cosine':
                match_matrix = tf.matmul(doc_emb, tf.transpose(query_emb, [0, 2, 1]))
                match_matrix /= tf.expand_dims(tf.sqrt(tf.reduce_sum(doc_emb * doc_emb, axis=2)), axis=2) * \
                                tf.expand_dims(tf.sqrt(tf.reduce_sum(query_emb * query_emb, axis=2)), axis=1)
    with vs.variable_scope('SelectiveJump'):
        # max size of location_ta and state_ta is max_jump_step + 1
        location_ta = tf.TensorArray(dtype=tf.float32, size=1, name='location_ta',
                                     clear_after_read=False, dynamic_size=True) # (d_ind,q_ind,d_len,q_len)
        location_ta = location_ta.write(0, tf.zeros([bs, 4])) # start from the top-left corner
        state_ta = tf.TensorArray(dtype=tf.float32, size=1, name='state_ta',
                                  clear_after_read=False, dynamic_size=True)
        if represent == 'sum_hard' or represent == 'test' or represent == 'rnn_hard':
            state_ta = state_ta.write(0, tf.zeros([bs, 1]))
        elif represent == 'cnn_hard':
            state_ta = state_ta.write(0, tf.zeros([bs, 1]))
        else:
            raise NotImplementedError()
        step = tf.zeros([bs], dtype=tf.int32)
        total_offset = tf.zeros([bs], dtype=tf.float32)
        is_stop = tf.zeros([bs], dtype=tf.bool)
        time = tf.constant(0)
        def cond(time, is_stop, step, state_ta, location_ta, dq_size, total_offset):
            return tf.logical_and(tf.logical_not(tf.reduce_all(is_stop)),
                                  tf.less(time, tf.constant(max_jump_step)))
        def body(time, is_stop, step, state_ta, location_ta, dq_size, total_offset):
            cur_location = location_ta.read(time)
            #time = tf.Print(time, [time], message='time:')
            with vs.variable_scope('Glimpse'):
                glimpse_location = get_glimpse_location(match_matrix, dq_size, cur_location, glimpse)
                # stop when the start index overflow
                new_stop = tf.reduce_any(glimpse_location[:, :2] > tf.cast(dq_size - 1, tf.float32), axis=1)
                glimpse_location = tf.where(new_stop, cur_location, glimpse_location)
                is_stop = tf.logical_or(is_stop, new_stop)
            with vs.variable_scope('Jump'):
                new_location = get_jump_location(match_matrix, dq_size, glimpse_location, jump, min_density=min_density)
                if max_jump_offset != None:
                    new_location = tf.concat([new_location[:, :2],
                                              tf.minimum(new_location[:, 2:], max_jump_offset)], axis=1)
                # stop when the start index overflow
                new_stop = tf.reduce_any(new_location[:, :2] > tf.cast(dq_size - 1, tf.float32), axis=1)
                is_stop = tf.logical_or(is_stop, new_stop)
                location_ta = location_ta.write(time + 1, tf.where(is_stop, cur_location, new_location))
                # total length to be modeled
                total_offset += tf.where(is_stop, tf.zeros_like(total_offset), new_location[:, 2])
                # actual rnn length (with padding)
                #total_offset += tf.where(is_stop, tf.zeros_like(total_offset), 
                #    tf.ones_like(total_offset) * \
                #    tf.reduce_max(tf.where(is_stop, tf.zeros_like(total_offset), new_location[:, 2])))
            with vs.variable_scope('Represent'):
                # location_one_out is to prevent duplicate time-consuming calculation
                location_one_out = tf.where(is_stop, tf.ones_like(cur_location), location_ta.read(time + 1))
                new_repr = get_representation(match_matrix, dq_size, query, query_emb, doc, doc_emb, word_vector, \
                                              location_one_out, represent, max_jump_offset=max_jump_offset, \
                                              doc_dpool_index=doc_dpool_index, query_dpool_index=query_dpool_index, \
                                              rnn_size=rnn_size, keep_prob=keep_prob)
                state_ta = state_ta.write(time + 1, tf.where(is_stop, state_ta.read(time), new_repr))
            step = step + tf.where(is_stop, tf.zeros([bs], dtype=tf.int32), tf.ones([bs], dtype=tf.int32))
            return time + 1, is_stop, step, state_ta, location_ta, dq_size, total_offset
        _, is_stop, step, state_ta, location_ta, dq_size, total_offset = \
            tf.while_loop(cond, body, [time, is_stop, step, state_ta, location_ta, dq_size, total_offset], 
                          parallel_iterations=1)
    with vs.variable_scope('Aggregate'):
        states = state_ta.stack()
        location = location_ta.stack()
        location = tf.transpose(location, [1, 0 ,2])
        stop_ratio = tf.reduce_mean(tf.cast(is_stop, tf.float32))
        complete_ratio = tf.reduce_mean(tf.reduce_min(
            [(location[:, -1, 0] + location[:, -1, 2]) / tf.cast(dq_size[:, 0], dtype=tf.float32),
             tf.ones([bs], dtype=tf.float32)], axis=0))
        if aggregate == 'max':
            signal = tf.reduce_max(states, 0)
        elif aggregate == 'sum':
            signal = tf.reduce_sum(states, 0) - states[-1] * \
                tf.cast(tf.expand_dims(time - step, axis=-1), dtype=tf.float32)
        return signal, {'step': step, 'location': location, 'match_matrix': match_matrix, 
                        'complete_ratio': complete_ratio, 'stop_ratio': stop_ratio,
                        'doc_emb': doc_emb, 'total_offset': total_offset}