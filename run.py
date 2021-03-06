import argparse, logging, os, random, time, json, functools, pickle
from itertools import groupby
import numpy as np
from tensorflow.python import debug as tf_debug
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from utils import Vocab, WordVector, load_prep_file, load_train_test_file, printoptions, \
  load_judge_file, NullContextManager
from metric import evaluate, ndcg, average_precision, precision
from rri import rri, DELTA
from cnn import DynamicMaxPooling, CNNVis

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='run')
  parser.add_argument('-a', '--action', help='action', type=str, default='train_test')
  parser.add_argument('-c', '--config', help='config file', type=str, default=None)
  parser.add_argument('--max_q_d_len', help='max query and doc length considered', type=str,
    default='10:1000')
  parser.add_argument('-D', '--debug', help='whether to use debug log level', action='store_true')
  parser.add_argument('--disable_gpu', help='whether to disable GPU', action='store_true')
  parser.add_argument('--gpu', help='which GPU to use', type=str, default='0')
  parser.add_argument('-d', '--data_dir', help='data directory', type=str)
  parser.add_argument('-w', '--word_vector_path', help='fila path to word vector', type=str, 
    default='w2v')
  parser.add_argument('-B', '--tf_summary_path', help='path to save tf summary', type=str, default=None)
  parser.add_argument('-s', '--save_model_path', help='path to save tf model', type=str, default=None)
  parser.add_argument('-r', '--reuse_model_path', help='path to the model to reuse', type=str, default=None)
  parser.add_argument('-f', '--format', help='format of input data. \
    "ir" for original format and "text" for new text matching format', type=str, default='ir')
  parser.add_argument('--reverse', help='whether to reverse the pairs in training testing files', 
    action='store_true')
  parser.add_argument('--tfrecord', help='whether to use tfrecord as input pipeline', 
    action='store_true')
  parser.add_argument('--no_normalize_w2v', help='whether to normalize w2v or not', action='store_true')
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
if not args.debug:
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    start_time = time.time()
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
        yield yield_result, time.time() - start_time
        start_time = time.time()
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
  NOT_FIT_EXCEPTION = 'need fit to be called before prediction'
  DECISION_EXCEPTION = 'prediction under classification loss function with >2 \
    relevance level is not support'

  def __init__(self, max_q_len=0, max_d_len=0, max_jump_step=0, word_vector=None, oov_word_vector=None,
         vocab=None, word_vector_trainable=True, use_pad_word=True, 
         interaction='dot', glimpse='fix_hard', glimpse_fix_size=None, min_density=None, use_ratio=False, 
         min_jump_offset=1, jump='max_hard', represent='sum_hard', input_mu=None, separate=False,
         all_position=True, direction='unidirectional', aggregate='max',
         rnn_size=None, max_jump_offset=None, max_jump_offset2=None, rel_level=2, loss_func='regression', margin=1.0, 
         keep_prob=1.0, paradigm='pointwise', learning_rate=0.1, random_seed=0, 
         n_epochs=100, batch_size=100, batch_num=None, batcher=None, verbose=1, save_epochs=None, reuse_model=None, 
         save_model=None, summary_path=None, tfrecord=False, tfrecord_has_weight=False,
         tfrecord_has_segmentation=False, unsupervised=False, small_dataset_num=-1):
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
    self.input_mu = input_mu
    self.separate = separate
    self.all_position = all_position
    self.direction = direction
    self.aggregate = aggregate
    self.rnn_size = rnn_size
    self.max_jump_offset = max_jump_offset
    self.max_jump_offset2 = max_jump_offset2
    self.rel_level = rel_level
    self.loss_func = loss_func
    self.margin = margin
    self.keep_prob = keep_prob
    self.paradigm = paradigm
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
    self.tfrecord = tfrecord
    self.tfrecord_has_weight = tfrecord_has_weight # wether use doc/query weight or not
    self.tfrecord_has_segmentation = tfrecord_has_segmentation # wether use doc segmentation or not
    self.unsupervised = unsupervised
    self.match_matrix_focus_debug = False
    # Use a subset of the datasetis for fast development. -1 means using all.
    self.small_dataset_num = small_dataset_num


  @staticmethod
  def parse_tfexample_fn_pairwise(example_proto, max_q_len, max_d_len, has_weight, has_segmentation):
    '''
    Parse a pairwise record.
    '''
    feature_to_type = {
      'qid': tf.FixedLenFeature([], dtype=tf.string),
      'docid1': tf.FixedLenFeature([], dtype=tf.string),
      'docid2': tf.FixedLenFeature([], dtype=tf.string),
      'query': tf.VarLenFeature(dtype=tf.int64),
      'doc1': tf.VarLenFeature(dtype=tf.int64),
      'doc2': tf.VarLenFeature(dtype=tf.int64),
      'qlen': tf.FixedLenFeature([1], dtype=tf.int64),
      'doc1len': tf.FixedLenFeature([1], dtype=tf.int64),
      'doc2len': tf.FixedLenFeature([1], dtype=tf.int64),
      'label': tf.FixedLenFeature([1], dtype=tf.float32),
    }
    if has_weight:
      feature_to_type['query_weight'] = tf.VarLenFeature(dtype=tf.float32)
      feature_to_type['doc1_weight'] = tf.VarLenFeature(dtype=tf.float32)
      feature_to_type['doc2_weight'] = tf.VarLenFeature(dtype=tf.float32)
    if has_segmentation:
      feature_to_type['doc1_segmentation'] = tf.VarLenFeature(dtype=tf.int64)
      feature_to_type['doc2_segmentation'] = tf.VarLenFeature(dtype=tf.int64)
    parsed_features = tf.parse_single_example(example_proto, feature_to_type)
    parsed_features['query'] = tf.sparse_tensor_to_dense(parsed_features['query'])
    parsed_features['doc1'] = tf.sparse_tensor_to_dense(parsed_features['doc1'])
    parsed_features['doc2'] = tf.sparse_tensor_to_dense(parsed_features['doc2'])
    query = tf.stack([parsed_features['query'], parsed_features['query']])
    if has_weight:
      parsed_features['query_weight'] = tf.sparse_tensor_to_dense(parsed_features['query_weight'])
      parsed_features['doc1_weight'] = tf.sparse_tensor_to_dense(parsed_features['doc1_weight'])
      parsed_features['doc2_weight'] = tf.sparse_tensor_to_dense(parsed_features['doc2_weight'])
      query_weight = tf.stack([parsed_features['query_weight'], parsed_features['query_weight']])
    if has_segmentation:
      parsed_features['doc1_segmentation'] = tf.sparse_tensor_to_dense(parsed_features['doc1_segmentation'])
      parsed_features['doc2_segmentation'] = tf.sparse_tensor_to_dense(parsed_features['doc2_segmentation'])
    qid = tf.stack([parsed_features['qid'], parsed_features['qid']])
    d1l = tf.shape(parsed_features['doc1'])[0]
    d2l = tf.shape(parsed_features['doc2'])[0]
    max_doc_len = tf.maximum(d1l, d2l)
    doc = tf.stack([tf.pad(parsed_features['doc1'], [[0, max_doc_len-d1l]]),
      tf.pad(parsed_features['doc2'], [[0, max_doc_len-d2l]])], axis=0)
    if has_weight:
      doc_weight = tf.stack([tf.pad(parsed_features['doc1_weight'], [[0, max_doc_len-d1l]]),
        tf.pad(parsed_features['doc2_weight'], [[0, max_doc_len-d2l]])], axis=0)
    if has_segmentation:
      doc_seg = tf.stack([tf.pad(parsed_features['doc1_segmentation'], [[0, max_doc_len-d1l]]),
        tf.pad(parsed_features['doc2_segmentation'], [[0, max_doc_len-d2l]])], axis=0)
    docid = tf.stack([parsed_features['docid1'], parsed_features['docid2']])
    qd_size1 = tf.concat([parsed_features['qlen'], parsed_features['doc1len']], axis=0)
    qd_size2 = tf.concat([parsed_features['qlen'], parsed_features['doc2len']], axis=0)
    qd_size = tf.stack([qd_size1, qd_size2])
    relevance = tf.stack([parsed_features['label'], parsed_features['label']])
    query = query[:, :max_q_len]
    doc = doc[:, :max_d_len]
    if has_weight:
      query_weight = query_weight[:, :max_q_len]
      doc_weight = doc_weight[:, :max_d_len]
    else:
      query_weight = tf.zeros_like(query, dtype=tf.float32)
      doc_weight = tf.zeros_like(doc, dtype=tf.float32)
    if has_segmentation:
      doc_seg = doc_seg[:, :max_d_len]
    else:
      doc_seg = tf.zeros_like(doc)
    qd_size = tf.minimum(qd_size, [[max_q_len, max_d_len]])
    #return tf.data.Dataset.from_tensor_slices((query, doc, qd_size, relevance, qid, docid))
    return query, query_weight, doc, doc_weight, qd_size, relevance, qid, docid, doc_seg


  @staticmethod
  def flat_map_tensor(*args):
    return tf.data.Dataset.from_tensor_slices(args)


  @staticmethod
  def parse_tfexample_fn_pointwise(example_proto, max_q_len, max_d_len, has_weight, has_segmentation):
    '''
    Parse a pointwise record.
    '''
    feature_to_type = {
      'qid': tf.FixedLenFeature([], dtype=tf.string),
      'docid': tf.FixedLenFeature([], dtype=tf.string),
      'query': tf.VarLenFeature(dtype=tf.int64),
      'doc': tf.VarLenFeature(dtype=tf.int64),
      'qlen': tf.FixedLenFeature([1], dtype=tf.int64),
      'doclen': tf.FixedLenFeature([1], dtype=tf.int64),
      'label': tf.FixedLenFeature([1], dtype=tf.float32),
    }
    if has_weight:
      feature_to_type['query_weight'] = tf.VarLenFeature(dtype=tf.float32)
      feature_to_type['doc_weight'] = tf.VarLenFeature(dtype=tf.float32)
    if has_segmentation:
      feature_to_type['doc_segmentation'] = tf.VarLenFeature(dtype=tf.int64)
    parsed_features = tf.parse_single_example(example_proto, feature_to_type)
    parsed_features['query'] = tf.sparse_tensor_to_dense(parsed_features['query'])
    parsed_features['doc'] = tf.sparse_tensor_to_dense(parsed_features['doc'])
    if has_weight:
      parsed_features['query_weight'] = tf.sparse_tensor_to_dense(parsed_features['query_weight'])
      parsed_features['doc_weight'] = tf.sparse_tensor_to_dense(parsed_features['doc_weight'])
    if has_segmentation:
      parsed_features['doc_segmentation'] = tf.sparse_tensor_to_dense(parsed_features['doc_segmentation'])
    query = parsed_features['query']
    doc = parsed_features['doc']
    qd_size = tf.concat([parsed_features['qlen'], parsed_features['doclen']], axis=0)
    relevance = parsed_features['label']
    query = query[:max_q_len]
    doc = doc[:max_d_len]
    if has_weight:
      query_weight = parsed_features['query_weight'][:max_q_len]
      doc_weight = parsed_features['doc_weight'][:max_d_len]
    else:
      query_weight = tf.zeros_like(query, dtype=tf.float32)
      doc_weight = tf.zeros_like(doc, dtype=tf.float32)
    if has_segmentation:
      doc_seg = parsed_features['doc_segmentation'][:max_d_len]
    else:
      doc_seg = tf.zeros_like(doc)
    qd_size = tf.minimum(qd_size, [max_q_len, max_d_len])
    return query, query_weight, doc, doc_weight, qd_size, relevance, \
      parsed_features['qid'], parsed_features['docid'], doc_seg


  def build_graph(self):
    with vs.variable_scope('Input'):
      if not self.tfrecord:
        '''
        placeholder input
        '''
        # dynamic query length of shape (batch_size, max_q_len)
        self.query = tf.placeholder(tf.int32, shape=[None, None], name='query')
        self.query_weight = tf.placeholder(tf.int32, shape=[None, None], name='query_weight')
        # dynamic doc length of shape (batch_size, max_d_len)
        self.doc = tf.placeholder(tf.int32, shape=[None, None], name='doc')
        self.doc_weight = tf.placeholder(tf.int32, shape=[None, None], name='doc_weight')
        # (batch_size, 2), the first column is query length, the second is doc length
        self.qd_size = tf.placeholder(tf.int32, shape=[None, 2], name='query_doc_size')
        # relevance signal (only useful when using pointwise)
        self.relevance = tf.placeholder(tf.int32, shape=[None], name='relevance')
        # query id
        self.qid = tf.placeholder(tf.string, shape=[None], name='qid')
        # docid
        self.docid = tf.placeholder(tf.string, shape=[None], name='docid')
        # doc segmentation, 1 means split and 0 means following
        self.doc_seg = tf.placeholder(tf.int32, shape=[None, None], name='doc_seg')
      else:
        '''
        tfrecord input
        '''
        def data_transform(dataset, paradigm='pointwise', is_train=True):
          # Procedure: interleave, take, (shuffle), (repeat), map, batch. () only for training.
          # Difference between train and test is that shuffle and repeat are 
          # only applied to train dataset.
          if paradigm == 'pointwise':
            parse_fn = RRI.parse_tfexample_fn_pointwise
          elif paradigm == 'pairwise':
            parse_fn = RRI.parse_tfexample_fn_pairwise
          # Interleave
          if is_train:
            dataset = dataset.shuffle(buffer_size=1)
          dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=1, block_length=1)
          # Take
          if self.small_dataset_num > 0:
            # take different amount of samples for training and testing
            ratio = 1 if is_train else 0.25
            take_num = int(self.small_dataset_num * ratio)
            print('use small dataset of size {}'.format(take_num))
            dataset = dataset.take(take_num)
          # Shuffle
          if is_train:
            buffer_size = 100000
            if self.small_dataset_num > 0:
               buffer_size = min(buffer_size, self.small_dataset_num)
            print('shuffle training data with buffer {}'.format(buffer_size))
            dataset = dataset.shuffle(buffer_size=buffer_size)
          else:
            dataset = dataset
          # Repeat
          if is_train and self.batch_num:
            # If we go through all the samples using batch_num, we don't need to re-initialize 
            # iterator of train dataset, which means that we need the repeat the dataset.
            dataset = dataset.repeat()
          # Map
          if paradigm == 'pointwise':
            dataset = dataset.map(functools.partial(parse_fn, 
              max_q_len=self.max_q_len, max_d_len=self.max_d_len, 
              has_weight=self.tfrecord_has_weight,
              has_segmentation=self.tfrecord_has_segmentation), num_parallel_calls=4)
          elif paradigm == 'pairwise':
            dataset = dataset.map(functools.partial(parse_fn, 
              max_q_len=self.max_q_len, max_d_len=self.max_d_len, 
              has_weight=self.tfrecord_has_weight,
              has_segmentation=self.tfrecord_has_segmentation), num_parallel_calls=4)
          if paradigm == 'pairwise':
            # flatten pairwise samples
            dataset = dataset.flat_map(RRI.flat_map_tensor)
          # Batch
          dataset = dataset.padded_batch(self.batch_size, padded_shapes=dataset.output_shapes)
          dataset = dataset.prefetch(self.batch_size)
          return dataset
        # name pattern of the input tfrecord file
        self.tfrecord_pattern = tf.placeholder(tf.string, shape=[], 
          name='tfrecord_pattern')
        # (1) train_dataset is used for training. It could be pointwise of pairwise.
        # (2) test_dataset has the same format with train_dataset but is used for testing.
        #     The paradigm of test_dataset should be the same with train_data.
        # (3) decision_dataset is used for decision_function, like ranking generationg.
        #     It should always be pointwise.
        train_dataset = tf.data.TFRecordDataset.list_files(self.tfrecord_pattern)
        test_dataset = tf.data.TFRecordDataset.list_files(self.tfrecord_pattern)
        decision_dataset = tf.data.TFRecordDataset.list_files(self.tfrecord_pattern)
        train_dataset = data_transform(train_dataset, self.paradigm, is_train=True)
        test_dataset = data_transform(test_dataset, self.paradigm, is_train=False)
        decision_dataset = data_transform(decision_dataset, 'pointwise', is_train=False)
        #dataset_iterator = tf.data.Iterator.from_structure(
        #  train_dataset.output_types, train_dataset.output_shapes)
        #self.train_data_init_op = dataset_iterator.make_initializer(train_dataset)
        #self.test_data_init_op = dataset_iterator.make_initializer(test_dataset)
        #self.decision_data_init_op = dataset_iterator.make_initializer(decision_dataset)
        self.handle = tf.placeholder(tf.string, shape=[])
        dataset_iterator = tf.data.Iterator.from_string_handle(self.handle, 
          train_dataset.output_types, train_dataset.output_shapes)
        self.train_data_init_op = train_dataset.make_initializable_iterator()
        self.test_data_init_op = test_dataset.make_initializable_iterator()
        self.decision_data_init_op = decision_dataset.make_initializable_iterator()
        self.query, self.query_weight, self.doc, self.doc_weight, self.qd_size, \
          self.relevance, self.qid, self.docid, self.doc_seg = dataset_iterator.get_next()
        self.query = tf.cast(self.query, dtype=tf.int32)
        self.doc = tf.cast(self.doc, dtype=tf.int32)
        self.qd_size = tf.cast(self.qd_size, dtype=tf.int32)
        self.relevance = tf.squeeze(tf.cast(self.relevance, dtype=tf.int32))
        self.qid = tf.squeeze(self.qid)
        self.docid = tf.squeeze(self.docid)
        self.doc_seg = tf.cast(self.doc_seg, dtype=tf.int32) if self.tfrecord_has_segmentation else None
      self.keep_prob_ = tf.placeholder(tf.float32) # dropout prob
    with vs.variable_scope('InputProcessing'):
      print('word vector trainable: {}'.format(self.word_vector_trainable))
      self.word_vector_variable = tf.get_variable('word_vector', self.word_vector.shape,
        initializer=tf.constant_initializer(self.word_vector),
        trainable=self.word_vector_trainable)
      word_vector_variable = self.word_vector_variable
      if self.oov_word_vector != None:
        # Don't train OOV words which are all located at the end of the word_vector matrix.
        print('don\'t train {} OOV word vector'.format(self.oov_word_vector))
        word_vector_variable = tf.concat([word_vector_variable[:-self.oov_word_vector],
          tf.stop_gradient(word_vector_variable[-self.oov_word_vector:])], axis=0)
      if self.use_pad_word:
        # Prepend a zero embedding at the first position of word_vector_variable.
        # Note that all the word inds need to be added 1 to be consistent with this change.
        print('prepend a pad_word at the word vectors')
        word_vector_variable = tf.concat([tf.zeros_like(word_vector_variable[:1]),
          word_vector_variable], axis=0)
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
          word_vector=word_vector_variable, interaction=self.interaction, glimpse=self.glimpse,
          glimpse_fix_size=self.glimpse_fix_size, min_density=self.min_density, use_ratio=self.use_ratio,
          min_jump_offset=self.min_jump_offset, max_jump_offset2=self.max_jump_offset2,
          jump=self.jump, represent=self.represent, separate=self.separate, all_position=self.all_position,
          direction=self.direction, aggregate=self.aggregate, rnn_size=self.rnn_size,
          max_jump_offset=self.max_jump_offset, keep_prob=self.keep_prob_, 
          query_weight=self.query_weight, doc_weight=self.doc_weight, input_mu=self.input_mu,
          doc_seg=self.doc_seg)
    initializer = tf.constant_initializer(1) if self.unsupervised else None
    if self.loss_func == 'classification':
      with vs.variable_scope('ClassificationLoss'):
        logit_w = tf.get_variable('logit_weight', 
          shape=[self.outputs.get_shape()[1], self.rel_level], 
            initializer=tf.constant_initializer([[0.0] * (self.rel_level-1) + [1.0]]))
        logit_b = tf.get_variable('logit_bias', 
          shape=[self.rel_level], initializer=tf.constant_initializer())
        logits = tf.nn.bias_add(tf.matmul(self.outputs, logit_w), logit_b)
        # The larger the last logits, the higher the score. Use logits is more reliable than
        # using softmax_cross_entropy_with_logits because exp explodes easily.
        self.scores = logits[:, -1]
        #self.scores = -tf.nn.softmax_cross_entropy_with_logits(
        #  labels=tf.one_hot(tf.ones_like(self.relevance)*(self.rel_level-1), self.rel_level), 
        #  logits=logits)        
        self.losses = tf.nn.softmax_cross_entropy_with_logits(
          labels=tf.one_hot(self.relevance, self.rel_level), logits=logits)
        self.loss = tf.reduce_mean(self.losses)
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self.relevance, predictions=tf.argmax(logits,1))
    elif self.loss_func == 'regression':
      with vs.variable_scope('RegressionLoss'):
        score_w = tf.get_variable('score_weight', 
          shape=[self.outputs.get_shape()[1], 1], initializer=initializer)
        score_b = tf.get_variable('score_bias', shape=[1], initializer=tf.constant_initializer())
        self.scores = tf.squeeze(tf.nn.bias_add(tf.matmul(self.outputs, score_w), score_b))
        self.loss = tf.nn.l2_loss(self.scores - tf.cast(self.relevance, dtype=tf.float32))
        self.acc_op, self.acc = tf.no_op(), tf.no_op()
    elif self.loss_func == 'pairwise_margin':
      with vs.variable_scope('PairwiseMargin'):
        score_w = tf.get_variable('score_weight', 
          shape=[self.outputs.get_shape()[1], 1], initializer=initializer)
        score_b = tf.get_variable('score_bias', shape=[1], initializer=tf.constant_initializer())
        self.scores = tf.squeeze(tf.nn.bias_add(tf.matmul(self.outputs, score_w), score_b))
        pairwise_scores = tf.reshape(self.scores, [-1, 2])
        self.losses = tf.maximum(0.0, self.margin-pairwise_scores[:, 0]+pairwise_scores[:, 1])
        self.loss = tf.reduce_mean(self.losses)
        decision = tf.cast(pairwise_scores[:, 0] > pairwise_scores[:, 1], dtype=tf.int32)
        self.acc, self.acc_op = tf.metrics.accuracy(labels=tf.ones_like(decision), predictions=decision)
    self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    self.init_all_vars = tf.global_variables_initializer()
    self.init_all_vars_local = tf.local_variables_initializer()
    self.saver = tf.train.Saver()
    if self.summary_path != None:
      self.summaries = tf.summary.merge_all()
      if self.summaries is None:
        self.summaries = tf.no_op()
      self.train_writer = tf.summary.FileWriter(os.path.join(self.summary_path, 'train'), self.graph_)
    if self.match_matrix_focus_debug:
      self.saliency = tf.gradients(-self.loss, [self.rri_info['match_matrix']])[0]
      saliency_dist = tf.reshape(tf.abs(self.saliency), [bs, -1])
      saliency_dist = saliency_dist / (tf.reduce_sum(saliency_dist, axis=1, keep_dims=True) + DELTA)
      match_matrix_flattend = tf.reshape(self.rri_info['match_matrix'], [bs, -1])
      self.match_matrix_focus = tf.reduce_sum(
        saliency_dist * match_matrix_flattend, axis=1)
      self.match_matrix_focus_bins = []
      match_matrix_cur = tf.cast(tf.zeros_like(match_matrix_flattend), dtype=tf.bool)
      #for density_region in [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 100]:
      for density_region in [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 100]:
        match_matrix_cur = tf.logical_and(match_matrix_flattend <= density_region,
          tf.logical_not(match_matrix_cur))
        match_matrix_cur = tf.cast(match_matrix_cur, dtype=saliency_dist.dtype)
        match_matrix_focus_cur = tf.reduce_sum(saliency_dist * match_matrix_cur, axis=1) / \
          (tf.reduce_sum(match_matrix_cur, axis=1) + DELTA)
        self.match_matrix_focus_bins.append(match_matrix_focus_cur)
        match_matrix_cur = match_matrix_flattend <= density_region
      self.match_matrix_focus_bins = tf.stack(self.match_matrix_focus_bins)
    else:
      self.saliency = tf.no_op()
      self.match_matrix_focus = tf.no_op()
      self.match_matrix_focus_bins = tf.no_op()
    #self.test_wv_grad = tf.gradients(self.loss, [self.word_vector_variable])[0]
    #self.test_rnn_grad = tf.gradients(self.loss, [v for v in tf.global_variables() 
    #    if 'rnn' in v.name and 'kernel' in v.name and 'Adam' not in v.name])[0]


  def check_params(self):
    if 'cnn' in self.represent and self.max_jump_offset == None:
      raise ValueError('max_jump_offset must be set when cnn is used')
    if self.paradigm not in {'pointwise', 'pairwise'}:
      raise ValueError('paradigm not supported')
    if self.paradigm == 'pairwise':
      self.loss_func = 'pairwise_margin'
    if self.loss_func not in {'classification', 'regression', 'pairwise_margin'}:
      raise ValueError('loss_func not supported')
    if self.paradigm == 'pairwise' and self.batch_size % 2 != 0:
      raise ValueError('batch_size should be even in pairwise setting')


  def get_w2v(self):
    if not hasattr(self, 'session_'):
      raise AttributeError(RRI.NOT_FIT_EXCEPTION)
    return self.session_.run(self.word_vector_variable)


  def feed_dict_postprocess(self, fd, is_train=True):
    if not self.tfrecord:
      feed_dict = {self.query: fd['query'], self.doc: fd['doc'], self.qd_size: fd['qd_size'], 
        self.qid: fd['qid'], self.docid: fd['docid']}
      if 'relevance' in fd:
        feed_dict[self.relevance] = fd['relevance']
    else:
      feed_dict = fd
    if is_train:
      feed_dict[self.keep_prob_] = self.keep_prob
    else:
      feed_dict[self.keep_prob_] = 1.0
    return feed_dict


  def init_graph_session(self):
    if not hasattr(self, 'session_'):
      self.graph_ = tf.Graph()
      with self.graph_.as_default():
        tf.set_random_seed(self.random_seed)
        #with vs.variable_scope('RRI', initializer=
        #  tf.truncated_normal_initializer(mean=0.0, stddev=1e-1, dtype=tf.float32, 
        #    seed=self.random_seed)) as scope:
        with vs.variable_scope('RRI', initializer=
          tf.glorot_uniform_initializer(seed=self.random_seed)) as scope:
          self.build_graph()
      config = tf.ConfigProto()
      config.allow_soft_placement = True
      config.log_device_placement = False
      config.gpu_options.allow_growth = True
      self.session_ = tf.Session(graph=self.graph_, config=config)
      if self.reuse_model:  # read model from file
        print('load model from "{}"'.format(self.reuse_model))
        self.saver.restore(self.session_, self.reuse_model)
      else: # initialize model
        self.session_.run(self.init_all_vars)
      #self.session_ = tf_debug.LocalCLIDebugWrapperSession(self.session_)


  def fit_iterable(self, X, y=None):
    def batcher_wrapper():
      def tfrecord_batcher(batch_num=None):
        bind = 0
        while True:
          bind += 1
          yield None, 0
          if batch_num and bind >= batch_num:
            break
      if not self.tfrecord:
        return self.batcher(X, y, self.batch_size, use_permutation=True, batch_num=self.batch_num)
      else:
        return tfrecord_batcher(batch_num=self.batch_num)
    trace_op, trace_graph = False, False
    # check params
    self.check_params()
    # init graph and session
    self.init_graph_session()
    #for i in self.graph_.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RRI'):
    #  print(i, i.eval(self.session_))
    #input('continue')
    # profile
    builder = option_builder.ProfileOptionBuilder
    #profiler = model_analyzer.Profiler(graph=self.graph_)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # train
    if self.unsupervised:
      yield
      return
    with open('match_matrix_focus_test', 'w') if self.match_matrix_focus_debug \
      else NullContextManager(None) as mmf_fout:
      for epoch in range(self.n_epochs):
        epoch += 1
        start = time.time()
        feed_time_all = 0
        loss_list, com_r_list, stop_r_list, total_offset_list, step_list = [], [], [], [], []
        if self.tfrecord:
          train_handle = self.session_.run(self.train_data_init_op.string_handle())
          if epoch == 1 or self.batch_num is None:
            # When batch_num is used, we don't need to re-initialize the dataset.
            self.session_.run(self.train_data_init_op.initializer, 
              feed_dict={self.tfrecord_pattern: X})
        # Local variables (total and count in accuracy operations) 
        # should be initialized every epoch
        self.session_.run(self.init_all_vars_local)
        #for i, (fd, feed_time) in enumerate(self.batcher(X, y, self.batch_size, use_permutation=True, batch_num=self.batch_num)):
        for i, (fd, feed_time) in enumerate(batcher_wrapper()):
          feed_time_all += feed_time
          if not self.tfrecord:
            batch_size = len(fd['query'])
          else:
            fd = {self.handle: train_handle}
            batch_size = None
          # tfrecord also need feed_dict, like keep_prob
          feed_dict = self.feed_dict_postprocess(fd, is_train=True)
          fetch = [self.rri_info['step'], self.rri_info['location'], self.rri_info['match_matrix'],
               self.loss, self.scores, self.rri_info['complete_ratio'], self.rri_info['is_stop'], self.rri_info['stop_ratio'], 
               self.rri_info['total_offset'], self.rri_info['signal'], self.rri_info['states'], self.rri_info['min_density'], 
               self.saliency, self.match_matrix_focus, self.match_matrix_focus_bins, self.trainer, 
               self.qid, self.docid, self.qd_size, self.relevance, self.doc, self.query]
          start_time = time.time()
          try:
            if self.summary_path != None and i % 10 == 0: # run statistics
              run_metadata = tf.RunMetadata()
              fetch += [self.summaries]
              step, location, match_matrix, loss, scores, com_r, is_stop, stop_r, total_offset, signal, states, \
              min_density, saliency, match_matrix_focus, match_matrix_focus_bins, _, fd['qid'], fd['docid'], \
              fd['qd_size'], fd['relevance'], fd['doc'], fd['query'], summaries = \
                self.session_.run(fetch, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
              global_step = (epoch - 1) * self.batch_num + i
              #print('adding summary for {}'.format(global_step))
              #self.train_writer.add_run_metadata(run_metadata, 'step%d' % global_step)
              if summaries is not None:
                self.train_writer.add_summary(summaries, global_step=global_step)
              if trace_op:
                profiler_opts = builder(builder.time_and_memory()).order_by('micros').build()
                tf.profiler.profile(self.graph_, run_meta=run_metadata, cmd='scope', options=profiler_opts)
                #input('press to continue')
              if trace_graph:
                #profiler.add_step(step=i, run_meta=run_metadata)
                #profiler_opts_builder = builder(builder.time_and_memory())
                #profiler_opts_builder.with_timeline_output(timeline_file='summary/profiler.json')
                #profiler_opts_builder.with_step(i)
                #profiler.profile_graph(profiler_opts_builder.build())
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open('summary/profiler.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format())
            else:
              step, location, match_matrix, loss, scores, com_r, is_stop, stop_r, total_offset, signal, states, \
              min_density, saliency, match_matrix_focus, match_matrix_focus_bins, _, fd['qid'], fd['docid'], \
              fd['qd_size'], fd['relevance'], fd['doc'], fd['query'] = \
                self.session_.run(fetch, feed_dict=feed_dict)
            if mmf_fout != None:
              for j, mmf in enumerate(match_matrix_focus):
                focus_bins = [match_matrix_focus_bins[k, j] \
                  for k in range(match_matrix_focus_bins.shape[0])]
                mmf_fout.write('{}\t{}\t{}\n'.format(mmf, scores[j], 
                  '\t'.join(map(lambda x: str(x), focus_bins))))
          except tf.errors.OutOfRangeError:
            if self.batch_num:
              raise Exception('Tfrecord OutOfRange is not expected because you are using batch_num')
            break
          end_time = time.time()
          loss_list.append(loss)
          com_r_list.append(com_r)
          stop_r_list.append(stop_r)
          total_offset_list.extend(total_offset)
          step_list.extend(step)
          if self.verbose >= 2:
            print('{:<5}\t{:>5.3f}\tloss:{:>5.3f}\tcom ratio:{:>3.2f}\tstop ratio:{:>3.2f}'
                .format(i, end_time - start_time, loss, com_r, stop_r))
          #for i in self.graph_.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RRI'):
          #  if i.name.startswith('RRI/InputProcessing/word_vector'):
          #    print(i, i.eval(self.session_))
          #input('continue')
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
            with printoptions(precision=6, suppress=True, threshold=np.nan):
              cnn_vis = CNNVis()
              batch_size = len(fd['qid'])
              print('match matrix size: {}'.format(match_matrix.shape))
              while True:
                b = input('which sample between 0 and {} (y for break)'.format(batch_size))
                if b == 'y':
                  break
                try:
                   b = int(b)
                except ValueError:
                   print('int or b')
                   continue
                print('qd_size: {}, step: {}, is_stop: {}, offset: {}'.format(
                  fd['qd_size'][b], step[b], is_stop[b], total_offset[b]))
                print('qid: {}, docid: {}, rel: {}, score: {}'.format(
                  fd['qid'][b], fd['docid'][b], fd['relevance'][b], scores[b]))
                # visualize jump location
                print(location[b, :step[b]+1])
                # visualize match_matrix
                #print(match_matrix[b][:fd['qd_size'][b, 1],:fd['qd_size'][b, 0]])
                #print(fd['doc'][b])
                print(np.max(match_matrix[b][:fd['qd_size'][b, 1], :fd['qd_size'][b, 0]], axis=1))
                print(np.max(match_matrix[b][:fd['qd_size'][b, 1], :fd['qd_size'][b, 0]], axis=0))
                #print(np.max(match_matrix[b], axis=1))
                #print(np.max(match_matrix[b], axis=0))
                # visualize match_matrix saliency (gradient)
                #saliency = np.abs(saliency)
                #top_s = np.argsort(-saliency[b].flatten())
                #print(saliency[b].flatten()[top_s[:10]])
                #print(match_matrix[b].flatten()[top_s[:10]])
                #print(min_density[b])
                #cnn_vis.plot_saliency_map(match_matrix, saliency)
                # visualize states
                print(states[1, b])
        if self.verbose >= 1:  # output epoch stat
          print('{:<10}\t{:>7}:{:>6.3f}\tstop:{:>5.3f}\toffset:{:>5.1f}\tstep:{:>3.1f}'
              .format('EPO[{}_{:>3.1f}_{:>3.1f}]'.format(epoch, (time.time() - start) / 60, feed_time_all/60),
                  'train', np.mean(loss_list), np.mean(stop_r_list), 
                  np.mean(total_offset_list), np.mean(step_list)), end='', flush=True)
        # save the model
        if self.save_epochs and (epoch % self.save_epochs == 0 or epoch == self.n_epochs):
          if self.save_model:
            self.saver.save(self.session_, self.save_model)
          yield
        if self.verbose:
          print('')


  def test(self, X, y=None):
    if not self.tfrecord:
      return self.test_placeholder(X, y);
    else:
      return self.test_tfrecord(X);


  def test_tfrecord(self, test_file_pattern):
    self.check_params()
    if not hasattr(self, 'session_'):
      raise AttributeError(RRI.NOT_FIT_EXCEPTION)
    loss_list, acc_list = [], []
    test_handle = self.session_.run(self.test_data_init_op.string_handle())
    self.session_.run(self.test_data_init_op.initializer, 
      feed_dict={self.tfrecord_pattern: test_file_pattern})
    self.session_.run(self.init_all_vars_local)
    while True:
      try:
        loss, _ = self.session_.run([self.loss, self.acc_op], 
          feed_dict={self.handle: test_handle})
        acc = self.session_.run(self.acc)
        acc_list.append(acc)
        loss_list.append(loss)
      except tf.errors.OutOfRangeError:
        break
    return np.mean(loss_list), acc_list[-1] if len(acc_list)>0 else np.nan


  def test_placeholder(self, X, y=None):
    self.check_params()
    if not hasattr(self, 'session_'):
      raise AttributeError(RRI.NOT_FIT_EXCEPTION)
    loss_list, acc_list = [], []
    self.session_.run(self.init_all_vars_local)
    for i, (fd, feed_time) in enumerate(self.batcher(X, y, self.batch_size, use_permutation=False)):
      feed_dict = self.feed_dict_postprocess(fd, is_train=False)
      loss, _ = self.session_.run([self.loss, self.acc_op], feed_dict=feed_dict)
      acc = self.session_.run(self.acc)
      acc_list.append(acc)
      loss_list.append(loss)
    return np.mean(loss_list), acc_list[-1] if len(acc_list)>0 else np.nan


  def get_ranking(self, q_list, doc_list, score_list):
    ranks = {}
    for q, dl in groupby(sorted(zip(q_list, doc_list, score_list), key=lambda x: x[0]), lambda x: x[0]):
      dl = sorted(dl, key=lambda x: -x[2])
      ranks[q] = [d[1] for d in dl]
    return ranks


  def decision_function(self, X, y=None):
    if not self.tfrecord:
      return self.decision_function_placeholder(X, y);
    else:
      return self.decision_function_tfrecord(X);


  def decision_function_tfrecord(self, decision_file_pattern):
    self.check_params()
    if not hasattr(self, 'session_'):
      raise AttributeError(RRI.NOT_FIT_EXCEPTION)
    q_list, doc_list, score_list, loss_list, acc_list = [], [], [], [], []
    # to investigate the density of match_matrix
    #rel_density, nonrel_density = [], []
    decision_handle = self.session_.run(self.decision_data_init_op.string_handle())
    self.session_.run(self.decision_data_init_op.initializer, 
      feed_dict={self.tfrecord_pattern: decision_file_pattern})
    self.session_.run(self.init_all_vars_local)
    while True:
      try:
        if self.loss_func == 'classification' and self.rel_level != 2:
          raise Exception(RRI.DECISION_EXCEPTION)
        if self.loss_func == 'pairwise_margin':
          '''
          In pairwise setting, the training graph (pairwise) and the ranking graph (pointwise) 
          are different. So loss and acc can not be accessed.
          '''
          scores, qid, docid, match_matrix, min_density, qd_size, relevance = \
            self.session_.run([self.scores, self.qid, self.docid, 
              self.rri_info['match_matrix'], self.rri_info['min_density'], self.qd_size, 
              self.relevance], feed_dict={self.handle: decision_handle})
        else:
          scores, loss, _, qid, docid, match_matrix, min_density, qd_size, relevance = \
            self.session_.run([self.scores, self.loss, self.acc_op, self.qid, self.docid, 
              self.rri_info['match_matrix'], self.rri_info['min_density'], self.qd_size, 
              self.relevance], feed_dict={self.handle: decision_handle})
          acc = self.session_.run(self.acc)
          acc_list.append(acc)
          loss_list.append(loss)
        #density = np.max(match_matrix, axis=2)
        #min_density = 0.3 * np.max(density, axis=1) + 0.7 * (np.sum(density, axis=1) / qd_size[:, 1])
        #min_density = np.percentile(density, 80, axis=1)
        #min_density = [np.percentile(density[i][:qd_size[i, 1]], 80) for i, d in enumerate(density)]
        #min_density = np.expand_dims(min_density, axis=1)
        #left_density = np.sum(density >= min_density, axis=1) / qd_size[:, 1]
        #for i, d in enumerate(left_density):
        #  if d >= 1:
        #    print(d, i, qd_size[i], density[i][:qd_size[i,1]], min_density[i, 0])
        #    input()
        #[rel_density.append(d) for i, d in enumerate(left_density) if relevance[i] > 0]
        #[nonrel_density.append(d) for i, d in enumerate(left_density) if relevance[i] <= 0]
        score_list.extend(scores)
        [q_list.append(q.decode('utf-8')) for q in qid]
        [doc_list.append(d.decode('utf-8')) for d in docid]
      except tf.errors.OutOfRangeError:
        break
    ranks = self.get_ranking(q_list, doc_list, score_list)
    #pickle.dump([rel_density, nonrel_density], open('density_test_tfpercentile_02.data', 'wb'))
    return ranks, np.mean(loss_list), acc_list[-1] if len(acc_list)>0 else np.nan


  def decision_function_placeholder(self, X, y=None):
    self.check_params()
    if not hasattr(self, 'session_'):
      raise AttributeError(RRI.NOT_FIT_EXCEPTION)
    q_list, doc_list, score_list, loss_list, acc_list = [], [], [], [], []
    self.session_.run(self.init_all_vars_local)
    for i, (fd, feed_time) in enumerate(self.batcher(X, y, self.batch_size, use_permutation=False)):
      feed_dict = self.feed_dict_postprocess(fd, is_train=False)
      if self.loss_func == 'classification' and self.rel_level != 2:
        raise Exception(RRI.DECISION_EXCEPTION)
      if self.loss_func == 'pairwise_margin':
        scores, = self.session_.run([self.scores], feed_dict=feed_dict)
      else:
        scores, loss, _ = self.session_.run([self.scores, self.loss, self.acc_op], 
          feed_dict=feed_dict)
        acc = self.session_.run(self.acc)
        acc_list.append(acc)
        loss_list.append(loss)
      score_list.extend(scores)
      [q_list.append(q) for q in fd['qid']]
      [doc_list.append(d) for d in fd['docid']]
    ranks = self.get_ranking(q_list, doc_list, score_list)
    return ranks, np.mean(loss_list), acc_list[-1] if len(acc_list)>0 else np.nan


def train_test():
  '''
  load config
  '''
  rel_level = 2
  def relevance_mapper(r):
    if r < 0:
      return 0
    if r >= rel_level:
      return rel_level - 1
    return r
  max_q_len_consider, max_d_len_consider = [int(l) for l in args.max_q_d_len.split(':')]  
  if args.config != None:
    model_config = json.load(open(args.config))
    print('model config: {}'.format(model_config))
  '''
  load word vector
  '''
  w2v_file = os.path.join(args.data_dir, args.word_vector_path)
  vocab_file = os.path.join(args.data_dir, 'vocab')
  print('loading word vector from {} ...'.format(w2v_file))
  wv = WordVector(filepath=w2v_file)
  vocab = Vocab(filepath=vocab_file, file_format=args.format)
  print('vocab size: {}, word vector dim: {}'.format(wv.vocab_size, wv.dim))
  '''
  test file path (used in both placeholder and tfrecord)
  '''
  test_file_judge = os.path.join(args.data_dir, 'test.prep.pointwise')
  if not args.tfrecord:
    '''
    load data (placeholder)
    '''
    train_file = os.path.join(args.data_dir, 'train.prep.{}'.format(args.paradigm))
    test_file = os.path.join(args.data_dir, 'test.prep.{}'.format(args.paradigm))
    doc_file = os.path.join(args.data_dir, 'docs.prep')
    if args.format == 'ir':
      query_file = os.path.join(args.data_dir, 'query.prep')
    print('loading query doc content ...')
    doc_raw = load_prep_file(doc_file, file_format=args.format)
    if args.format == 'ir':
      query_raw = load_prep_file(query_file, file_format=args.format)
      truncate_len = max_d_len_consider
    else:
      query_raw = doc_raw
      truncate_len = max(max_q_len_consider, max_d_len_consider)
    print('truncate long document')
    d_long_count = 0
    avg_doc_len, avg_truncate_doc_len = 0, 0
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
      test_X_judge, test_y_judge = test_X, test_y
    print('number of training samples: {}'.format(sum([len(x['query']) for x in train_X])))
  else:
    '''
    load data (tfrecord)
    '''
    train_X, train_y = os.path.join(args.data_dir, 
      'train.prep.{}.tfrecord-???-of-???'.format(args.paradigm)), None
    test_X, test_y = os.path.join(args.data_dir, 
      'test.prep.{}.tfrecord-???-of-???'.format(args.paradigm)), None
    test_X_judge, test_y_judge = os.path.join(args.data_dir, 
      'test.prep.pointwise.tfrecord-???-of-???'), None
    max_q_len = max_q_len_consider
    max_d_len = max_d_len_consider
    batcher = None
  '''
  load judge file
  '''
  test_qd_judge = load_judge_file(test_file_judge, file_format=args.format, reverse=args.reverse)
  for q in test_qd_judge:
    for d in test_qd_judge[q]:
      test_qd_judge[q][d] = relevance_mapper(test_qd_judge[q][d])

  '''
  train and test the model
  '''
  model_config_ = {
    'max_q_len': max_q_len, 
    'max_d_len': max_d_len, 
    'max_jump_step': 100, 
    'word_vector': wv.get_vectors(normalize=not args.no_normalize_w2v),
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
    'input_mu': None, 
    'separate': False,
    'all_position': False,
    'direction': 'unidirectional',
    'aggregate': 'max', 
    'rnn_size': 300, 
    'max_jump_offset': max_d_len, 
    'max_jump_offset2': max_q_len, 
    'rel_level': rel_level, 
    'loss_func': 'classification',
    'margin': 1.0,
    'keep_prob': 1.0, 
    'paradigm': args.paradigm,
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
    'summary_path': args.tf_summary_path,
    'tfrecord': args.tfrecord,
    'tfrecord_has_weight': False,
    'tfrecord_has_segmentation': False,
    'unsupervised': False,
    'small_dataset_num': -1,
  }
  if args.config != None:
    model_config_.update(model_config)
  rri = RRI(**model_config_)
  if not args.tfrecord:
    #train_X, train_y, test_X, test_y = train_X[:10], train_y[:10], test_X[:10], test_y[:10]
    #test_X_judge, test_y_judge = test_X_judge[:10], test_y_judge[:10]
    print('train query: {}, test query: {}'.format(len(train_X), len(test_X)))
  for i, e in enumerate(rri.fit_iterable(train_X, train_y)):
    i = i+1
    if i % 1 != 0:
      continue
    start = time.time()
    if args.format == 'ir':
      ranks, loss, acc = rri.decision_function(test_X_judge, test_y_judge)
      scores = evaluate(ranks, test_qd_judge, metric=ndcg, top_k=20)
      avg_score = np.mean(list(scores.values()))
    elif args.format == 'text':
      #loss, acc = rri.test(test_X, test_y)
      ranks, loss, acc = rri.decision_function(test_X_judge, test_y_judge)
      scores = evaluate(ranks, test_qd_judge, metric=average_precision, top_k=10000)
      avg_score = np.mean(list(scores.values()))
    #if not os.path.exists('ranking'):
    #  os.mkdir('ranking')
    #json.dump(ranks, open('ranking/ranking.{}.json'.format(i), 'w'))
    #if i % 1 == 0:
    #  w2v_update = rri.get_w2v()
    #  wv.update(w2v_update)
    #  wv.save_to_file('w2v_update')
    print('\t{:>7}:{:>5.3f}:{:>5.3f}:{:>5.3f}'
      .format('test_{:>3.1f}'.format((time.time()-start)/60), 
        loss, acc, avg_score), end='', flush=True)


if __name__ == '__main__':
  if args.action == 'train_test':
    train_test()
