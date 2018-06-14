import argparse, logging, os, random, time
from itertools import groupby
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from utils import Vocab, WordVector, load_prep_file, load_train_test_file, printoptions
from rri import rri

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run')
    parser.add_argument('-a', '--action', help='action', type=str, default='train_test')
    parser.add_argument('-D', '--debug', help='whether to use debug log level', action='store_true')
    parser.add_argument('-d', '--data_dir', help='data directory', type=str)
    parser.add_argument('-B', '--tf_summary_path', help='path to save tf summary', type=str, default=None)
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

SEED = 2018
random.seed(SEED)
np.random.seed(SEED)


def data_pad(samples, max_len, dtype=None):
    return np.array([s + [0] * (max_len - len(s)) for s in samples], dtype=dtype)


def data_assemble(filepath, query_raw, doc_raw, max_q_len, max_d_len):
    samples = load_train_test_file(filepath)
    samples_gb_q = groupby(samples, lambda x:x[0])
    X = []
    y = []
    if filepath.endswith('pointwise'):
        def batcher(X, y, batch_size):
            rb = batch_size
            result = {
                'qd_size': [],
                'relevance': [],
                'query': [],
                'doc': [],
            }
            query_ind = 0
            doc_ind = 0
            while query_ind < len(X):
                q_x = X[query_ind]
                q_y = y[query_ind]
                remain_n_sample = len(q_x['query']) - doc_ind
                take_n_sample = min(remain_n_sample, rb)
                for d in range(doc_ind, doc_ind + take_n_sample):
                    result['qd_size'].append(q_x['qd_size'][d])
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
                        'relevance': np.array(result['relevance'], dtype=np.int32),
                    }
                    yield_result['query'] = data_pad(result['query'], np.max(yield_result['qd_size'][:, 0]),
                                               np.int32)
                    yield_result['doc'] = data_pad(result['doc'], np.max(yield_result['qd_size'][:, 1]), np.int32)
                    yield yield_result
                    rb = batch_size
                    result = {
                        'qd_size': [],
                        'relevance': [],
                        'query': [],
                        'doc': [],
                    }
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
                q_x['query'].append(query_raw[s[0]])
                q_x['doc'].append(doc_raw[s[1]])
                q_x['qd_size'].append([len(query_raw[s[0]]), len(doc_raw[s[1]])])
                q_x['qid'].append(s[0])
                q_x['docid'].append(s[1])
                q_y['relevance'].append(s[2])
            X.append(q_x)
            y.append(q_y)
        return X, y, batcher
    else:
        raise NotImplementedError()


class RRI(object):
    INTERACTION = {'dot', 'cosine', 'indicator'}

    def __init__(self, max_q_len, max_d_len, max_jump_step, word_vector=None, interaction='dot',
                 glimpse='fix_hard', glimpse_fix_size=None, min_density=None, jump='max_hard', represent='sum_hard', rel_level=2,
                 learning_rate=0.1, random_seed=0, n_epochs=100, batch_size=100, batcher=None, verbose=True,
                 save_epochs=None, reuse_model=None, save_model=None, summary_path=None):
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len
        self.max_jump_step = max_jump_step
        self.word_vector = word_vector
        self.interaction = interaction
        self.glimpse = glimpse
        self.glimpse_fix_size = glimpse_fix_size
        self.min_density = min_density
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
        self.summary_path = summary_path


    def build_graph(self):
        with vs.variable_scope('Input'):
            self.query = tf.placeholder(tf.int32, shape=[None, None], name='query') # dynamic query length
            self.doc = tf.placeholder(tf.int32, shape=[None, None], name='doc') # dynamic doc length
            self.qd_size = tf.placeholder(tf.int32, shape=[None, 2], name='query_doc_size')
            self.relevance = tf.placeholder(tf.int32, shape=[None], name='relevance')
        self.outputs, self.rri_info = \
            rri(self.query, self.doc, tf.reverse(self.qd_size, [1]), max_jump_step=self.max_jump_step,
                word_vector=self.word_vector, interaction=self.interaction, glimpse=self.glimpse,
                glimpse_fix_size=self.glimpse_fix_size, min_density=self.min_density,
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
        if self.summary_path != None:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.summary_path, 'train'), self.graph_)


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
            config = tf.ConfigProto()
            #config = tf.ConfigProto(device_count={'CPU': 10}, intra_op_parallelism_threads=2)
            self.session_ = tf.Session(graph=self.graph_, config=config)
            self.session_.run(self.init_all_vars)
        # train
        for epoch in range(self.n_epochs):
            epoch += 1
            start = time.time()
            loss_list, = [],
            for i, fd in enumerate(self.batcher(X, y, self.batch_size)):
                batch_size = len(fd['query'])
                fetch = [self.rri_info['step'], self.rri_info['location'], self.rri_info['match_matrix'],
                         self.loss, self.trainer]
                feed_dict = {self.query: fd['query'], self.doc: fd['doc'], self.qd_size: fd['qd_size'],
                             self.relevance: fd['relevance']}
                start_time = time.time()
                if self.summary_path != None and i % 1 == 0: # run statistics
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    step, location, match_matrix, loss, _ = \
                        self.session_.run(fetch, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    self.train_writer.add_run_metadata(run_metadata, 'step%d' % i)
                    print('adding run metadata for', i)
                else:
                    step, location, match_matrix, loss, _ = self.session_.run(fetch, feed_dict=feed_dict)
                loss_list.append(loss)
                print(i, time.time() - start_time, loss,
                      np.mean([fd['qd_size'][b, 1] / (step[b] or 1) for b in range(batch_size)]))
                if args.debug:
                    with printoptions(precision=3, suppress=True, threshold=np.nan):
                        for b in range(batch_size):
                            print(fd['qd_size'][b, 1], step[b])
                            print(location[b])
                            print(np.max(match_matrix[b][:fd['qd_size'][b, 1], :], axis=1))
                            input()
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


def train_test():
    train_file = os.path.join(args.data_dir, 'train.prep.pointwise')
    test_file = os.path.join(args.data_dir, 'test.prep.pointwise')
    query_file = os.path.join(args.data_dir, 'query.prep')
    doc_file = os.path.join(args.data_dir, 'docs.prep')
    w2v_file = os.path.join(args.data_dir, 'w2v')
    vocab_file = os.path.join(args.data_dir, 'vocab')
    print('loading word vector ...')
    wv = WordVector(filepath=w2v_file)
    vocab = Vocab(filepath=vocab_file)
    print('loading query doc content ...')
    query_raw = load_prep_file(query_file)
    doc_raw = load_prep_file(doc_file)
    max_q_len = max([len(query_raw[q]) for q in query_raw])
    max_d_len = max([len(doc_raw[d]) for d in doc_raw])
    print('data assemble with max_q_len: {}, max_d_len: {} ...'.format(max_q_len, max_d_len))
    train_X, train_y, batcher = data_assemble(train_file, query_raw, doc_raw, max_q_len, max_d_len)
    print('number of samples: {}'.format(sum([len(x['query']) for x in train_X])))
    rri = RRI(max_q_len=max_q_len, max_d_len=max_d_len, max_jump_step=100, word_vector=wv.get_vectors(normalize=True),
              interaction='dot', glimpse='all_next', glimpse_fix_size=10, min_density=0.5,
              jump='min_density_hard', represent='sum_hard', rel_level=5, learning_rate=0.1, random_seed=SEED,
              n_epochs=10, batch_size=1000, batcher=batcher, verbose=True, save_epochs=None,
              reuse_model=None, save_model=None, summary_path=args.tf_summary_path)
    list(rri.fit_iterable(train_X, train_y))


if __name__ == '__main__':
    if args.action == 'train_test':
        train_test()