import argparse, logging, os, random, time, json
from itertools import groupby
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from utils import Vocab, WordVector, load_prep_file, load_train_test_file, printoptions, load_judge_file
from metric import evaluate, ndcg
from rri import rri
from cnn import DynamicMaxPooling

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run')
    parser.add_argument('-a', '--action', help='action', type=str, default='train_test')
    parser.add_argument('-c', '--config', help='config file', type=str, default=None)
    parser.add_argument('-D', '--debug', help='whether to use debug log level', action='store_true')
    parser.add_argument('--disable_gpu', help='whether to disable GPU', action='store_true')
    parser.add_argument('--gpu', help='which GPU to use', type=str, default='0')
    parser.add_argument('-d', '--data_dir', help='data directory', type=str)
    parser.add_argument('-B', '--tf_summary_path', help='path to save tf summary', type=str, default=None)
    parser.add_argument('-s', '--save_model_path', help='path to save tf model', type=str, default=None)
    parser.add_argument('-r', '--reuse_model_path', help='path to the model to reuse', type=str, default=None)
    parser.add_argument('-f', '--format', help='format of input data. \
        "ir" for original format and "text" for new text matching format', type=str, default='ir')
    parser.add_argument('--reverse', help='whether to reverse the pairs in training testing files', 
        action='store_true')
    parser.add_argument('-p', '--paradigm', help='learning to rank paradigm', type=str, 
        default='pointwise')
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.disable_gpu:
    print('diable GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline

SEED = 2018
random.seed(SEED)
np.random.seed(SEED)


def data_pad(samples, max_len, dtype=None):
    return np.array([s + [0] * (max_len - len(s)) for s in samples], dtype=dtype)


def data_assemble(filepath, query_raw, doc_raw, max_q_len, max_d_len, relevance_mapper=None):
    relevance_mapper = relevance_mapper or (lambda x: x)
    samples = load_train_test_file(filepath, file_format=args.format, reverse=args.reverse)
    samples_gb_q = groupby(samples, lambda x: x[0]) # queries should be sorted
    X = []
    y = []
    def batcher(X, y=None, batch_size=128, use_permutation=True, batch_num=None):
        rb = batch_size
        result = {
            'qid': [],
            'docid': [],
            'qd_size': [],
            'relevance': [],
            'query': [],
            'doc': [],
        }
        query_ind = 0
        doc_ind = 0
        total_batch_num = 0
        if use_permutation:
            # permutation wrt query
            perm = np.random.permutation(len(X))
        else:
            perm = list(range(len(X)))
        while query_ind < len(X):
            q_x = X[perm[query_ind]]
            q_y = y[perm[query_ind]] if y != None else None
            remain_n_sample = len(q_x['query']) - doc_ind
            take_n_sample = min(remain_n_sample, rb)
            for d in range(doc_ind, doc_ind + take_n_sample):
                result['qid'].append(q_x['qid'][d])
                result['docid'].append(q_x['docid'][d])
                result['qd_size'].append(q_x['qd_size'][d])
                if q_y != None:
                    result['relevance'].append(q_y['relevance'][d])
                result['query'].append(q_x['query'][d])
                result['doc'].append(q_x['doc'][d])
            rb -= take_n_sample
            doc_ind += take_n_sample
            if rb > 0 or doc_ind >= len(q_x['query']):
                query_ind += 1
                doc_ind = 0
            if rb == 0 or (len(result['qd_size']) > 0 and query_ind >= len(X)):
                # return batch
                yield_result = {
                    'qd_size': np.array(result['qd_size'], dtype=np.int32),
                }
                if q_y != None:
                    yield_result['relevance'] = np.array(result['relevance'], dtype=np.int32)
                yield_result['query'] = data_pad(result['query'], np.max(yield_result['qd_size'][:, 0]),
                                           np.int32)
                yield_result['doc'] = data_pad(result['doc'], np.max(yield_result['qd_size'][:, 1]), np.int32)
                yield_result['qid'] = np.array(result['qid'], dtype=str)
                yield_result['docid'] = np.array(result['docid'], dtype=str)
                #print('qid: {}'.format(list(zip(range(len(result['qid'])), result['qid']))))
                #print('docid: {}'.format(list(zip(range(len(result['docid'])), result['docid']))))
                total_batch_num += 1
                yield yield_result
                if batch_num and total_batch_num >= batch_num:
                    # end the batcher without traverse all the samples
                    break
                rb = batch_size
                result = {
                    'qid': [],
                    'docid': [],
                    'qd_size': [],
                    'relevance': [],
                    'query': [],
                    'doc': [],
                }
    if filepath.endswith('pointwise'):
        for q, q_samples in samples_gb_q:
            q_x = {
                'query': [],
                'doc': [],
                'qd_size': [],
                'max_q_len': max_q_len,
                'max_d_len': max_d_len,
                'qid': [],
                'docid': [],
            }
            q_y = {
                'relevance': [],
            }
            for s in q_samples:
                # use max_q_len and max_d_len to filter the queries and documents
                qm = query_raw[s[0]][:max_q_len]
                dm = doc_raw[s[1]][:max_d_len]
                q_x['query'].append(qm)
                q_x['doc'].append(dm)
                q_x['qd_size'].append([len(qm), len(dm)])
                q_x['qid'].append(s[0])
                q_x['docid'].append(s[1])
                q_y['relevance'].append(relevance_mapper(s[2]))
            X.append(q_x)
            y.append(q_y)
        return X, y, batcher
    elif filepath.endswith('pairwise'):
        for q, q_samples in samples_gb_q:
            q_x = {
                'query': [],
                'doc': [],
                'qd_size': [],
                'max_q_len': max_q_len,
                'max_d_len': max_d_len,
                'qid': [],
                'docid': [],
            }
            q_y = {
                'relevance': [],
            }
            for s in q_samples:
                # use max_q_len and max_d_len to filter the queries and documents
                if s[3] == 0:
                    # only consider pairs with difference
                    continue
                qm = query_raw[s[0]][:max_q_len]
                dm1 = doc_raw[s[1]][:max_d_len]
                dm2 = doc_raw[s[2]][:max_d_len]
                q_x['query'].append(qm)
                q_x['query'].append(qm)
                q_x['qid'].append(s[0])
                q_x['qid'].append(s[0])
                if s[3] < 0:
                    # only use positive pairs
                    dm = dm1
                    dm1 = dm2
                    dm2 = dm
                    q_x['docid'].append(s[2])
                    q_x['docid'].append(s[1])
                    q_y['relevance'].append(-s[2])
                    q_y['relevance'].append(-s[2])
                else:
                    q_x['docid'].append(s[1])
                    q_x['docid'].append(s[2])
                    q_y['relevance'].append(s[2])
                    q_y['relevance'].append(s[2])
                q_x['doc'].append(dm1)
                q_x['doc'].append(dm2)
                q_x['qd_size'].append([len(qm), len(dm1)])
                q_x['qd_size'].append([len(qm), len(dm2)])
            X.append(q_x)
            y.append(q_y)
            def pairwise_batcher(X, y=None, batch_size=128, use_permutation=True, batch_num=None):
                if batch_size % 2 != 0:
                    raise Exception('this batcher can\'t be used in pairwise approach')
                return batcher(X, y=y, batch_size=batch_size, use_permutation=use_permutation, batch_num=batch_num)
        return X, y, pairwise_batcher
    else:
        raise NotImplementedError()


class RRI(object):
    INTERACTION = {'dot', 'cosine', 'indicator'}

    def __init__(self, max_q_len=0, max_d_len=0, max_jump_step=0, word_vector=None, oov_word_vector=None,
                 vocab=None, word_vector_trainable=True, use_pad_word=True, 
                 interaction='dot', glimpse='fix_hard', glimpse_fix_size=None, min_density=None, use_ratio=False, 
                 min_jump_offset=1, jump='max_hard', represent='sum_hard', separate=False, aggregate='max', rnn_size=None, 
                 max_jump_offset=None, max_jump_offset2=None, rel_level=2, loss_func='regression', keep_prob=1.0, 
                 learning_rate=0.1, random_seed=0, 
                 n_epochs=100, batch_size=100, batch_num=None, batcher=None, verbose=1, save_epochs=None, reuse_model=None, 
                 save_model=None, summary_path=None):
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len
        self.max_jump_step = max_jump_step
        self.word_vector = word_vector
        self.oov_word_vector = oov_word_vector
        self.vocab = vocab
        self.word_vector_trainable = word_vector_trainable
        self.use_pad_word = use_pad_word
        self.interaction = interaction
        self.glimpse = glimpse
        self.glimpse_fix_size = glimpse_fix_size
        self.min_density = min_density
        self.use_ratio = use_ratio
        self.min_jump_offset = min_jump_offset
        self.jump = jump
        self.represent = represent
        self.separate = separate
        self.aggregate = aggregate
        self.rnn_size = rnn_size
        self.max_jump_offset = max_jump_offset
        self.max_jump_offset2 = max_jump_offset2
        self.rel_level = rel_level
        self.loss_func = loss_func
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.random_seed= random_seed
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.batcher = batcher
        self.verbose = verbose
        self.save_epochs = save_epochs
        self.reuse_model = reuse_model
        self.save_model = save_model
        self.summary_path = summary_path


    def build_graph(self):
        with vs.variable_scope('Input'):
            self.query = tf.placeholder(tf.int32, shape=[None, None], name='query') # dynamic query length
            self.doc = tf.placeholder(tf.int32, shape=[None, None], name='doc') # dynamic doc length
            self.qd_size = tf.placeholder(tf.int32, shape=[None, 2], name='query_doc_size')
            self.relevance = tf.placeholder(tf.int32, shape=[None], name='relevance')
            self.keep_prob_ = tf.placeholder(tf.float32) # dropout prob
            self.word_vector_variable = tf.get_variable('word_vector', self.word_vector.shape,
                                                        initializer=tf.constant_initializer(self.word_vector),
                                                        trainable=self.word_vector_trainable)
            if self.oov_word_vector != None:
                # Don't train OOV words which are all located at the end of the word_vector matrix.
                print('don\'t train {} OOV word vector'.format(self.oov_word_vector))
                self.word_vector_variable = tf.concat([self.word_vector_variable[:-self.oov_word_vector],
                    tf.stop_gradient(self.word_vector_variable[-self.oov_word_vector:])], axis=0)
            if self.use_pad_word:
                # Prepend a zero embedding at the first position of word_vector_variable.
                # Note that all the word inds need to be added 1 to be consistent with this change.
                print('prepend a pad_word at the word vectors')
                self.word_vector_variable = tf.concat([tf.zeros_like(self.word_vector_variable[:1]),
                    self.word_vector_variable], axis=0)
                bs = tf.shape(self.doc)[0]
                max_q_len = tf.shape(self.query)[1]
                max_d_len = tf.shape(self.doc)[1]
                query_pad = tf.expand_dims(tf.range(max_q_len), dim=0) < self.qd_size[:, :1]
                doc_pad = tf.expand_dims(tf.range(max_d_len), dim=0) < self.qd_size[:, 1:]
                query = self.query + tf.cast(query_pad, dtype=tf.int32)
                doc = self.doc + tf.cast(doc_pad, dtype=tf.int32)
            else:
                query = self.query
                doc = self.doc
        with vs.variable_scope('Arch'):
            self.outputs, self.rri_info = \
                rri(query, doc, tf.reverse(self.qd_size, [1]), max_jump_step=self.max_jump_step,
                    word_vector=self.word_vector_variable, interaction=self.interaction, glimpse=self.glimpse,
                    glimpse_fix_size=self.glimpse_fix_size, min_density=self.min_density, use_ratio=self.use_ratio,
                    min_jump_offset=self.min_jump_offset, max_jump_offset2=self.max_jump_offset2,
                     jump=self.jump, represent=self.represent, 
                    separate=self.separate, aggregate=self.aggregate, rnn_size=self.rnn_size, 
                    max_jump_offset=self.max_jump_offset, keep_prob=self.keep_prob_)
        if self.loss_func == 'classification':
            with vs.variable_scope('ClassificationLoss'):
                logit_w = tf.get_variable('logit_weight', shape=[self.outputs.get_shape()[1], self.rel_level])
                logit_b = tf.get_variable('logit_bias', shape=[self.rel_level], initializer=tf.constant_initializer())
                logits = tf.nn.bias_add(tf.matmul(self.outputs, logit_w), logit_b)
                self.losses = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(self.relevance, self.rel_level), logits=logits)
                self.loss = tf.reduce_mean(self.losses)
                self.acc, self.acc_op = tf.metrics.accuracy(labels=self.relevance, predictions=tf.argmax(logits,1))
        elif self.loss_func == 'regression':
            with vs.variable_scope('RegressionLoss'):
                score_w = tf.get_variable('score_weight', shape=[self.outputs.get_shape()[1], 1])
                score_b = tf.get_variable('score_bias', shape=[1], initializer=tf.constant_initializer())
                self.scores = tf.squeeze(tf.nn.bias_add(tf.matmul(self.outputs, score_w), score_b))
                self.loss = tf.nn.l2_loss(self.scores - tf.cast(self.relevance, dtype=tf.float32))
        elif self.loss_func == 'pairwise_margin':
            with vs.variable_scope('PairwiseMargin'):
                score_w = tf.get_variable('score_weight', shape=[self.outputs.get_shape()[1], 1])
                score_b = tf.get_variable('score_bias', shape=[1], initializer=tf.constant_initializer())
                self.scores = tf.squeeze(tf.nn.bias_add(tf.matmul(self.outputs, score_w), score_b))
                pairwise_scores = tf.reshape(self.scores, [-1, 2])
                self.losses = tf.maximum(0.0, 1 - pairwise_scores[:, 0] + pairwise_scores[:, 1])
                self.loss = tf.reduce_mean(self.losses)
                decision = tf.cast(pairwise_scores[:, 0] > pairwise_scores[:, 1], dtype=tf.int32)
                self.acc, self.acc_op = tf.metrics.accuracy(labels=tf.ones_like(decision), predictions=decision)
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.init_all_vars = tf.global_variables_initializer()
        self.init_all_vars_local = tf.local_variables_initializer()
        self.saver = tf.train.Saver()
        if self.summary_path != None:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.summary_path, 'train'), self.graph_)
        #self.test_wv_grad = tf.gradients(self.loss, [self.word_vector_variable])[0]
        #self.test_rnn_grad = tf.gradients(self.loss, [v for v in tf.global_variables() 
        #    if 'rnn' in v.name and 'kernel' in v.name and 'Adam' not in v.name])[0]


    def check_params(self):
        if self.reuse_model and not hasattr(self, 'session_'):  # read model from file
            self.graph_ = tf.Graph()
            with self.graph_.as_default():
                tf.set_random_seed(self.random_seed)
                with vs.variable_scope('RRI') as scope:
                    self.build_graph()
                    #scope.reuse_variables()
                    #self.build_graph_test()
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            #config = tf.ConfigProto()
            self.session_ = tf.Session(graph=self.graph_, config=config)
            logging.info('load model from "{}"'.format(self.reuse_model))
            self.saver.restore(self.session_, self.reuse_model)
        if 'cnn' in self.represent and self.max_jump_offset == None:
            raise ValueError('max_jump_offset must be set when cnn is used')
        if self.loss_func not in {'classification', 'regression', 'pairwise_margin'}:
            raise ValueError('loss_func not supported')


    def feed_dict_postprocess(self, fd, is_train=True):
        feed_dict = {self.query: fd['query'], self.doc: fd['doc'], self.qd_size: fd['qd_size'],
                     self.relevance: fd['relevance']}
        if is_train:
            feed_dict[self.keep_prob_] = self.keep_prob
        else:
            feed_dict[self.keep_prob_] = 1.0
        return feed_dict


    def fit_iterable(self, X, y=None):
        trace_op, trace_graph = True, False
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
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            #config = tf.ConfigProto()
            #device_count={'GPU': 0}
            self.session_ = tf.Session(graph=self.graph_, config=config)
            self.session_.run(self.init_all_vars)
            self.session_.run(self.init_all_vars_local)
        # profile
        builder = option_builder.ProfileOptionBuilder
        #profiler = model_analyzer.Profiler(graph=self.graph_)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        # train
        for epoch in range(self.n_epochs):
            epoch += 1
            start = time.time()
            loss_list, com_r_list, stop_r_list, total_offset_list, step_list = [], [], [], [], []
            for i, fd in enumerate(self.batcher(X, y, self.batch_size, use_permutation=True, batch_num=self.batch_num)):
                batch_size = len(fd['query'])
                fetch = [self.rri_info['step'], self.rri_info['location'], self.rri_info['match_matrix'],
                         self.loss, self.rri_info['complete_ratio'], self.rri_info['is_stop'], self.rri_info['stop_ratio'], 
                         self.rri_info['total_offset'], self.trainer]
                feed_dict = self.feed_dict_postprocess(fd, is_train=True)
                start_time = time.time()
                if self.summary_path != None and i % 1 == 0: # run statistics
                    step, location, match_matrix, loss, com_r, is_stop, stop_r, total_offset, _ = \
                        self.session_.run(fetch, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    end_time = time.time()
                    self.train_writer.add_run_metadata(run_metadata, 'step%d' % i)
                    print('adding run metadata for {}'.format(i))
                    if trace_op:
                        profiler_opts = builder(builder.time_and_memory()).order_by('micros').build()
                        tf.profiler.profile(self.graph_, run_meta=run_metadata, cmd='op', options=profiler_opts)
                        input('press to continue')
                    if trace_graph:
                        #profiler.add_step(step=i, run_meta=run_metadata)
                        #profiler_opts_builder = builder(builder.time_and_memory())
                        #profiler_opts_builder.with_timeline_output(timeline_file='summary/profiler.json')
                        #profiler_opts_builder.with_step(i)
                        #profiler.profile_graph(profiler_opts_builder.build())
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open('summary/profiler.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format())
                    print('profile run metadata for {}'.format(i))
                else:
                    step, location, match_matrix, loss, com_r, is_stop, stop_r, total_offset, _ = \
                        self.session_.run(fetch, feed_dict=feed_dict)
                    end_time = time.time()
                loss_list.append(loss)
                com_r_list.append(com_r)
                stop_r_list.append(stop_r)
                [total_offset_list.append(to) for to in total_offset]
                [step_list.append(st) for st in step]
                if self.verbose >= 2:
                    print('{:<5}\t{:>5.3f}\tloss:{:>5.3f}\tcom ratio:{:>3.2f}\tstop ratio:{:>3.2f}'
                          .format(i, end_time - start_time, loss, com_r, stop_r))
                if args.debug and hasattr(self, 'test_rnn_grad'):
                    test_rnn_grad, = self.session_.run([self.test_rnn_grad], feed_dict=feed_dict)
                    with printoptions(precision=3, suppress=True, threshold=np.nan):
                        print(np.max(np.abs(test_rnn_grad), axis=1))
                        #input()
                if args.debug and self.word_vector_trainable and hasattr(self, 'test_wv_grad'):
                    test_grad, = self.session_.run([self.test_wv_grad], feed_dict=feed_dict)
                    with printoptions(precision=3, suppress=True, threshold=np.nan):
                        print(type(test_grad))
                        print(len(test_grad.indices))
                        #print(np.sum(np.any(test_grad[0] != 0, axis=2), axis=1))
                        inds = np.any(np.abs(test_grad.values) >= .001, axis=1)
                        w_inds, u_w_inds = test_grad.indices[inds], np.unique(test_grad.indices[inds])
                        print(np.sum(w_inds), len(u_w_inds))
                        print(w_inds)
                        print(u_w_inds)
                        print(self.vocab.decode(u_w_inds))
                        print(np.max(np.abs(test_grad.values[inds]), axis=1))
                        #input()
                if args.debug:
                    cont = input('continue debug? y for yes:')
                    if cont != 'y':
                        continue
                    with printoptions(precision=3, suppress=True, threshold=np.nan):
                        for b in range(batch_size):
                            print('qd_size: {}, step: {}, is_stop: {}, offset: {}'.format(
                                fd['qd_size'][b], step[b], is_stop[b], total_offset[b]))
                            print('qid: {}, docid: {}'.format(fd['qid'][b], fd['docid'][b]))
                            print(location[b, :step[b]+1])
                            print(np.max(match_matrix[b][:fd['qd_size'][b, 1], :fd['qd_size'][b, 0]], axis=1))
                            bcont = input('break? y for yes:')
                            if bcont == 'y':
                                break
            if self.verbose >= 1:  # output epoch stat
                print('{:<10}\t{:>7}:{:>6.3f}\tstop:{:>5.3f}\toffset:{:>5.1f}\tstep:{:>3.1f}'
                      .format('EPO[{}_{:>3.1f}]'.format(epoch, (time.time() - start) / 60),
                              'train', np.mean(loss_list), np.mean(stop_r_list), 
                              np.mean(total_offset_list), np.mean(step_list)), end='', flush=True)
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


    def test(self, X, y=None):
        self.check_params()
        if not hasattr(self, 'session_'):
            raise AttributeError('need fit or fit_iterable to be called before prediction')
        loss_list = []
        acc_list = []
        for i, fd in enumerate(self.batcher(X, y, self.batch_size, use_permutation=False)):
            feed_dict = self.feed_dict_postprocess(fd, is_train=False)
            if self.loss_func in {'classification', 'pairwise_margin'}:
                loss, _, acc = self.session_.run([self.loss, self.acc_op, self.acc], feed_dict=feed_dict)
                acc_list.append(acc)
            else:
                loss, = self.session_.run([self.loss], feed_dict=feed_dict)
            loss_list.append(loss)
        return np.mean(loss_list), np.mean(acc_list)


    def decision_function(self, X):
        self.check_params()
        if not hasattr(self, 'session_'):
            raise AttributeError('need fit or fit_iterable to be called before prediction')
        q_list, doc_list, score_list, loss_list = [], [], [], []
        for i, fd in enumerate(self.batcher(X, None, self.batch_size, use_permutation=False)):
            fd['relevance'] = np.ones([fd['query'].shape[0]], dtype=np.int32) * (self.rel_level - 1)
            feed_dict = self.feed_dict_postprocess(fd, is_train=False)
            if self.loss_func == 'classification':
                if self.rel_level != 2:
                    raise Exception('prediction under classification loss function with >2 \
                        relevance level is not support')
                losses, = self.session_.run([self.losses], feed_dict=feed_dict)
                [loss_list.append(l) for l in losses]
                [score_list.append(-l) for l in losses]
                [q_list.append(q) for q in fd['qid']]
                [doc_list.append(d) for d in fd['docid']]
            elif self.loss_func in {'regression', 'pairwise_margin'}:
                scores, loss, = self.session_.run([self.scores, self.loss], feed_dict=feed_dict)
                loss_list.append(loss)
                [score_list.append(s) for s in scores]
                [q_list.append(q) for q in fd['qid']]
                [doc_list.append(d) for d in fd['docid']]
        ranks = {}
        for q, dl in groupby(sorted(zip(q_list, doc_list, score_list), key=lambda x: x[0]), lambda x: x[0]):
            dl = sorted(dl, key=lambda x: -x[2])
            ranks[q] = [d[1] for d in dl]
        return ranks, np.mean(loss_list)


def train_test():
    if args.config != None:
        model_config = json.load(open(args.config))
        print('model config: {}'.format(model_config))
    train_file = os.path.join(args.data_dir, 'train.prep.{}'.format(args.paradigm))
    test_file = os.path.join(args.data_dir, 'test.prep.{}'.format(args.paradigm))
    test_file_judge = os.path.join(args.data_dir, 'test.prep.pointwise')
    if args.format == 'ir':
        query_file = os.path.join(args.data_dir, 'query.prep')
    doc_file = os.path.join(args.data_dir, 'docs.prep')
    w2v_file = os.path.join(args.data_dir, 'w2v')
    vocab_file = os.path.join(args.data_dir, 'vocab')
    rel_level = 2
    max_q_len_consider = 10
    max_d_len_consider = 1000
    print('loading word vector ...')
    wv = WordVector(filepath=w2v_file)
    vocab = Vocab(filepath=vocab_file, file_format=args.format)
    print('vocab size: {}, word vector dim: {}'.format(wv.vocab_size, wv.dim))
    print('loading query doc content ...')
    doc_raw = load_prep_file(doc_file, file_format=args.format)
    if args.format == 'ir':
        query_raw = load_prep_file(query_file, file_format=args.format)
    else:
        query_raw = doc_raw
    print('truncate long document')
    d_long_count = 0
    avg_doc_len, avg_truncate_doc_len = 0, 0
    truncate_len = max(max_q_len_consider, max_d_len_consider)
    for d in doc_raw:
        avg_doc_len += len(doc_raw[d])
        if len(doc_raw[d]) > truncate_len:
            d_long_count += 1
            doc_raw[d] = doc_raw[d][:truncate_len]
            avg_truncate_doc_len += truncate_len
        else:
            avg_truncate_doc_len += len(doc_raw[d])
    avg_doc_len = avg_doc_len / len(doc_raw)
    avg_truncate_doc_len = avg_truncate_doc_len / len(doc_raw)
    print('total doc: {}, long doc: {}, average len: {}, average truncate len: {}'.format(
        len(doc_raw), d_long_count, avg_doc_len, avg_truncate_doc_len))
    max_q_len = min(max_q_len_consider, max([len(query_raw[q]) for q in query_raw]))
    max_d_len = min(max_d_len_consider, max([len(doc_raw[d]) for d in doc_raw]))
    print('data assemble with max_q_len: {}, max_d_len: {} ...'.format(max_q_len, max_d_len))
    def relevance_mapper(r):
        if r < 0:
            return 0
        if r >= rel_level:
            return rel_level - 1
        return r
    train_X, train_y, batcher = data_assemble(train_file, query_raw, doc_raw, max_q_len, max_d_len, 
                                              relevance_mapper=relevance_mapper)
    '''
    doc_len_list = []
    for q_x in train_X:
        for d in q_x['qd_size']:
            doc_len_list.append(d[1])
    doc_len_list = np.array(doc_len_list, dtype=np.int32)
    doc_len_list = [min(max_jump_offset ** 2 / d, max_jump_offset) for d in doc_len_list]
    plt.hist(doc_len_list, bins=max_jump_offset)
    plt.xlim(xmin=0, xmax=max_jump_offset)
    plt.xlabel('preserve number')
    plt.ylabel('number')
    plt.show()
    '''
    test_X, test_y, _ = data_assemble(test_file, query_raw, doc_raw, max_q_len, max_d_len, 
                                      relevance_mapper=relevance_mapper)
    if args.paradigm == 'pairwise':
        test_X_judge, test_y_judge, _ = data_assemble(test_file_judge, query_raw, doc_raw, max_q_len, max_d_len, 
                                          relevance_mapper=relevance_mapper)
    else:
        text_X_judge, test_y_judge = test_X, test_y
    test_qd_judge = load_judge_file(test_file_judge, file_format=args.format, reverse=args.reverse)
    for q in test_qd_judge:
        for d in test_qd_judge[q]:
            test_qd_judge[q][d] = relevance_mapper(test_qd_judge[q][d])
    print('number of training samples: {}'.format(sum([len(x['query']) for x in train_X])))
    model_config_ = {
        'max_q_len': max_q_len, 
        'max_d_len': max_d_len, 
        'max_jump_step': 100, 
        'word_vector': wv.get_vectors(normalize=True),
        'oov_word_vector': None,
        'vocab': vocab, 
        'word_vector_trainable': False,
        'use_pad_word': True, 
        'interaction': 'dot', 
        'glimpse': 'all_next_hard', 
        'glimpse_fix_size': 10,
        'min_density': -1, 
        'use_ratio': False, 
        'min_jump_offset': 3, 
        'jump': 'min_density_hard', 
        'represent': 'interaction_cnn_hard', 
        'separate': False, 
        'aggregate': 'max', 
        'rnn_size': 16, 
        'max_jump_offset': 50, 
        'max_jump_offset2': max_q_len, 
        'rel_level': rel_level, 
        'loss_func': 'classification',
        'keep_prob': 1.0, 
        'learning_rate': 0.0002, 
        'random_seed': SEED, 
        'n_epochs': 30, 
        'batch_size': 256,
        'batch_num': 400, 
        'batcher': batcher, 
        'verbose': 1, 
        'save_epochs': 1, 
        'reuse_model': args.reuse_model_path, 
        'save_model': args.save_model_path, 
        'summary_path': args.tf_summary_path
    }
    if args.paradigm == 'pairwise':
        model_config_['loss_func'] = 'pairwise_margin'
    if args.config != None:
        model_config_.update(model_config)
    rri = RRI(**model_config_)
    #train_X, train_y, test_X, test_y = train_X[:2560], train_y[:2560], test_X[:2560], test_y[:2560]
    print('train query: {}, test query: {}'.format(len(train_X), len(test_X)))
    for e in rri.fit_iterable(train_X, train_y):
        start = time.time()
        loss, acc = rri.test(test_X, test_y)
        if args.format == 'ir':
            ranks, _ = rri.decision_function(test_X_judge)
            scores = evaluate(ranks, test_qd_judge, metric=ndcg, top_k=20)
            avg_score = np.mean(list(scores.values()))
        elif args.format == 'text':
            avg_score = None
        print('\t{:>7}:{:>5.3f}:{:>5.3f}:{:>5.3f}'
            .format('test_{:>3.1f}'.format((time.time()-start)/60), 
                loss, acc, avg_score), end='', flush=True)


if __name__ == '__main__':
    if args.action == 'train_test':
        train_test()
