import logging, time, sys
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs


def rri(query, doc, dq_size, max_jump_step, word_vector, interaction='dot', glimpse='fix_hard', glimpse_fix_size=None,
        jump='max_hard', represent='sum_hard'):
    bs = tf.shape(query)[0]
    max_q_len = query.get_shape().as_list()[1]
    max_d_len = doc.get_shape().as_list()[1]
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
    with vs.variable_scope('Jump'):
        if jump.find('max') != -1:
            # glimpse by looking forward 'glimpse_fix_size' cells
            match_matrix_pad = \
                tf.pad(match_matrix, tf.constant([[0, 0], [0, glimpse_fix_size], [0, 0]]), 'CONSTANT',
                       constant_values=sys.float_info.min)
        else:
            match_matrix_pad = match_matrix
        # max size of location_ta and state_ta is max_jump_step + 1
        location_ta = tf.TensorArray(dtype=tf.float32, size=1, name='location_ta',
                                     clear_after_read=False, dynamic_size=True) # (d_ind,q_ind,d_len,q_len)
        location_ta = location_ta.write(0, tf.zeros([bs, 4])) # start from the top-left corner
        state_ta = tf.TensorArray(dtype=tf.float32, size=1, name='state_ta',
                                  clear_after_read=False, dynamic_size=True)
        if represent == 'sum_hard':
            state_ta = state_ta.write(0, tf.zeros([bs, 1]))
        else:
            raise NotImplementedError()
        step = tf.zeros([bs], dtype=tf.int32)
        is_stop = tf.zeros([bs], dtype=tf.bool)
        time = tf.constant(0)
        def get_glimpse_location(match_matrix_pad, dq_size, location):
            if glimpse == 'fix_hard':
                gp_d_position = tf.cast(tf.floor(location[:, 0] + location[:, 2]), dtype=tf.int32)
                gp_d_offset = tf.reduce_min([tf.ones_like(dq_size[:, 0], dtype=tf.int32) * glimpse_fix_size,
                                             dq_size[:, 0] - gp_d_position], axis=0)
                return tf.stack([tf.cast(gp_d_position, dtype=tf.float32), location[:, 1],
                                 tf.cast(gp_d_offset, dtype=tf.float32), location[:, 3]], axis=1)
        '''
        def get_glimpse_region(match_matrix_pad, dq_size, location):
            if glimpse == 'fix_hard':
                gp_ind_center = tf.cast(tf.floor(location[:, 0]), dtype=tf.int32) + \
                                tf.range(bs) * (max_d_len + glimpse_fix_size)
                gp_ind_region = tf.reshape(
                    tf.stack([gp_ind_center + i for i in range(0, glimpse_fix_size + 1)], axis=1), [-1])
                glimpse_region = tf.reshape(tf.gather(tf.reshape(match_matrix_pad, [-1, max_q_len]), gp_ind_region),
                                            [-1, glimpse_fix_size + 1, max_q_len])
            else:
                raise NotImplementedError()
            return glimpse_region
        '''
        def get_jump_location(match_matrix_pad, dq_size, location):
            if jump == 'max_hard':
                d_len = tf.shape(match_matrix_pad)[1]
                max_d_offset = tf.cast(tf.floor(tf.reduce_max(location[:, 2])), dtype=tf.int32)
                gp_ind_center = tf.cast(tf.floor(location[:, 0]), dtype=tf.int32) + tf.range(bs) * d_len
                gp_ind_region = tf.reshape(tf.expand_dims(gp_ind_center, axis=-1) +
                                           tf.expand_dims(tf.range(max_d_offset), axis=0), [-1])
                glimpse_region = tf.reshape(tf.gather(tf.reshape(match_matrix_pad, [-1, max_q_len]), gp_ind_region),
                                            [-1, max_d_offset, max_q_len])
                d_loc = tf.cast(tf.argmax(tf.reduce_max(tf.abs(glimpse_region), axis=2), axis=1), dtype=tf.float32) + \
                        location[:, 0]
                new_location = tf.stack([d_loc, location[:, 1], tf.ones([bs]), location[:, 3]], axis=1)
            else:
                raise NotImplementedError()
            return new_location
        '''
        def get_jump_location(glimpse_region, dq_size, location):
            if jump == 'max_hard':
                d_loc = tf.cast(
                    tf.argmax(tf.reduce_max(tf.abs(glimpse_region), axis=2), axis=1), dtype=tf.float32) + \
                        location[:, 0]
                new_location = tf.stack([d_loc, location[:, 1], tf.ones([bs]), tf.ones([bs])], axis=1)
            else:
                raise NotImplementedError()
            return new_location
        '''
        def get_representation(match_matrix_pad, dq_size, query_emb, doc_emb, location):
            if represent == 'sum_hard':
                start = tf.cast(tf.floor(location[:, :2]), dtype=tf.int32)
                end = tf.cast(tf.floor(location[:, :2] + location[:, 2:]), dtype=tf.int32)
                ind = tf.constant(0)
                representation_ta = tf.TensorArray(dtype=tf.float32, size=bs,
                                                   name='representation_ta', clear_after_read=False)
                def body(i, m, s, e, r):
                    r_i = tf.reduce_sum(m[i][s[i, 0]:e[i, 0]+1])
                    r = r.write(i, tf.reshape(r_i, [1]))
                    return i + 1, m, s, e, r
                _, _, _, _, representation_ta = \
                    tf.while_loop(lambda i, m, s, e, r: i < bs, body,
                                  [ind, match_matrix_pad, start, end, representation_ta],
                                  parallel_iterations=1)
                representation = representation_ta.stack()
            else:
                raise NotImplementedError()
            return representation
        def cond(time, is_stop, step, state_ta, location_ta, dq_size):
            return tf.logical_and(tf.logical_not(tf.reduce_all(is_stop)),
                                  tf.less(time, tf.constant(max_jump_step)))
        def body(time, is_stop, step, state_ta, location_ta, dq_size):
            cur_location = location_ta.read(time)
            glimpse_location = get_glimpse_location(match_matrix_pad, dq_size, cur_location)
            new_location = get_jump_location(match_matrix_pad, dq_size, glimpse_location)
            new_stop = tf.reduce_any(new_location[:, :2] > tf.cast(dq_size, tf.float32), axis=1) # stop when the start index overflow
            is_stop = tf.logical_or(is_stop, new_stop)
            location_ta = location_ta.write(time + 1, tf.where(is_stop, cur_location, new_location))
            new_repr = get_representation(match_matrix_pad, dq_size, query_emb, doc_emb, location_ta.read(time + 1))
            state_ta = state_ta.write(time + 1, tf.where(is_stop, state_ta.read(time), new_repr))
            step = step + tf.where(is_stop, tf.zeros([bs], dtype=tf.int32), tf.ones([bs], dtype=tf.int32))
            return time + 1, is_stop, step, state_ta, location_ta, dq_size
        _, is_stop, step, state_ta, location_ta, dq_size = \
            tf.while_loop(cond, body, [time, is_stop, step, state_ta, location_ta, dq_size], parallel_iterations=1)
    with vs.variable_scope('Aggregate'):
        states = state_ta.stack()
        location = location_ta.stack()
        location = tf.transpose(location, [1, 0 ,2])
        return tf.reduce_max(states, 0), {'step': step, 'location': location, 'match_matrix': match_matrix}



class RRI(object):
    INTERACTION = {'dot', 'cosine', 'indicator'}

    def __init__(self, max_q_len, max_d_len, max_jump_step, word_vector=None, interaction='dot',
                 glimpse='fix_hard', glimpse_fix_size=None, jump='max_hard', represent='sum_hard', rel_level=2,
                 learning_rate=0.1, random_seed=0, n_epochs=100, batch_size=100, batcher=None, verbose=True,
                 save_epochs=None, reuse_model=None, save_model=None):
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len
        self.max_jump_step = max_jump_step
        self.word_vector = word_vector
        self.interaction = interaction
        self.glimpse = glimpse
        self.glimpse_fix_size = glimpse_fix_size
        self.jump = jump
        self.represent = represent
        self.rel_level = rel_level
        self.learning_rate = learning_rate
        self.random_seed= random_seed
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.batcher = batcher
        self.verbose = verbose
        self.save_epochs = save_epochs
        self.reuse_model = reuse_model
        self.save_model = save_model


    def build_graph(self):
        with vs.variable_scope('Input'):
            self.query = tf.placeholder(tf.int32, shape=[None, self.max_q_len], name='query')
            self.doc = tf.placeholder(tf.int32, shape=[None, self.max_d_len], name='doc')
            self.qd_size = tf.placeholder(tf.int32, shape=[None, 2], name='query_doc_size')
            self.relevance = tf.placeholder(tf.int32, shape=[None], name='relevance')
        self.outputs, self.rri_info = \
            rri(self.query, self.doc, tf.reverse(self.qd_size, [1]), max_jump_step=self.max_jump_step, word_vector=self.word_vector,
                interaction=self.interaction, glimpse=self.glimpse, glimpse_fix_size=self.glimpse_fix_size,
                jump=self.jump, represent=self.represent)
        with vs.variable_scope('Loss'):
            logit_w = tf.get_variable('logits_weight', shape=[self.outputs.get_shape()[1], self.rel_level])
            logit_b = tf.get_variable('logits_bias', shape=[self.rel_level], initializer=tf.constant_initializer())
            logits = tf.nn.bias_add(tf.matmul(self.outputs, logit_w), logit_b)
            self.loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.relevance, self.rel_level), logits=logits)
            self.loss = tf.reduce_mean(self.loss)
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.init_all_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    def check_params(self):
        if self.reuse_model and not hasattr(self, 'session_'):  # read model from file
            self.graph_ = tf.Graph()
            with self.graph_.as_default():
                tf.set_random_seed(self.random_seed)
                with vs.variable_scope('RRI') as scope:
                    self.build_graph()
                    #scope.reuse_variables()
                    #self.build_graph_test()
            self.session_ = tf.Session(graph=self.graph_)
            logging.info('load model from "{}"'.format(self.reuse_model))
            self.saver.restore(self.session_, self.reuse_model)


    def fit_iterable(self, X, y=None):
        # check params
        self.check_params()
        # init graph and session
        if not hasattr(self, 'session_'):
            self.graph_ = tf.Graph()
            with self.graph_.as_default():
                tf.set_random_seed(self.random_seed)
                with vs.variable_scope('RRI') as scope:
                    self.build_graph()
                    #scope.reuse_variables()
                    #self.build_graph_test()
            self.session_ = tf.Session(graph=self.graph_)
            self.session_.run(self.init_all_vars)
        # train
        for epoch in range(self.n_epochs):
            epoch += 1
            start = time.time()
            loss_list, = [],
            for i, fd in enumerate(self.batcher(X, y, self.batch_size)):
                step, location, match_matrix, loss, _ = self.session_.run(
                    [self.rri_info['step'], self.rri_info['location'], self.rri_info['match_matrix'],
                     self.loss, self.trainer],
                    feed_dict={self.query: fd['query'], self.doc: fd['doc'],
                               self.qd_size: fd['qd_size'], self.relevance: fd['relevance']})
                print(loss)
                #print(step[:5])
                #print(location[:5])
                #for b in range(5):
                #    print(np.max(match_matrix[b][:15, :], axis=1))
                #input()
                loss_list.append(loss)
            if self.verbose:  # output epoch stat
                print('{:<10}\t{:>7}:{:>5.3f}:{:>7.3f}'
                      .format('EPO[{}_{:>3.1f}]'.format(epoch, (time.time() - start) / 60),
                              'train', np.mean([]), np.mean(loss_list)), end='')
            if self.save_epochs and epoch % self.save_epochs == 0:  # save the model
                if self.save_model:
                    self.saver.save(self.session_, self.save_model)
                yield
            # save the final model
            elif epoch == self.n_epochs and self.save_epochs and self.n_epochs % self.save_epochs != 0:
                if self.save_model:
                    self.saver.save(self.session_, self.save_model)
                yield
            if self.verbose:
                print('')