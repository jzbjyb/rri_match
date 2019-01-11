import os, argparse, logging, jpype, random, gzip, shutil, zlib, multiprocessing, signal, time, pickle
from collections import defaultdict
from itertools import groupby
import numpy as np
from jpype import *
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import Vocab, WordVector, load_from_html, load_from_query_file, load_train_test_file, \
  save_train_test_file, load_judge_file, load_run_file, load_query_log, save_judge_file, \
  save_query_file, load_prep_file, load_prep_file_aslist, save_prep_file, word_segment, qc_xml_field_line_map, \
  load_boilerpipe, qd_xml_iterator, prep_file_iterator, prep_file_mapper, PointwiseSample, qd_xml_to_prep, \
  qd_xml_filter
from config import CONFIG

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='preprocess')
  parser.add_argument('-a', '--action', help='action', type=str, default=None)
  parser.add_argument('-m', '--max_vocab_size', help='max vocabulary size', type=int, default=50000)
  parser.add_argument('-t', '--num_thread', help='number of threads', type=int, default=8)
  parser.add_argument('--line_count', help='number of lines of a file', type=int, default=None)
  parser.add_argument('-d', '--data_dir', help='data directory', type=str)
  parser.add_argument('-o', '--out_dir', help='output directory', type=str)
  parser.add_argument('-l', '--query_log', help='the filepath of query log', type=str)
  parser.add_argument('-r', '--train_test_ratio', help='the ratio of train and test dataset',
            type=float, default=0.8)
  parser.add_argument('-w', '--word_vector_path', help='the filepath of word vector', type=str)
  parser.add_argument('--first_line', help='whether the word vector file has the first line', 
    action='store_true')
  parser.add_argument('-D', '--debug', help='whether to use debug log level', action='store_true')
  parser.add_argument('--shuqi_bing_web_dir', help='shuqi\'s html dir', type=str)
  parser.add_argument('--min_query_freq', help='minimum query frequency', type=int, default=100)
  parser.add_argument('--judgement_refer', help='judgment referred to', type=str)
  parser.add_argument('-f', '--format', help='format of input data. \
    "ir" for original format and "text" for new text matching format', type=str, default='ir')
  parser.add_argument('--reverse', help='whether to reverse the pairs in training testing files', 
    action='store_true')
  parser.add_argument('--use_stream', help='whether to use streaming algorithm',
    action='store_true')
  parser.add_argument('--binary_html', help='whether to read html in binary', action='store_true')
  parser.add_argument('--gzip_files', help='filepath of gzip files', type=str)
  parser.add_argument('-p', '--paradigm', help='learning to rank paradigm', type=str, 
    default='pointwise')
  args = parser.parse_args()
  if args.debug:
    logging.basicConfig(level=logging.DEBUG)
  else:
    logging.basicConfig(level=logging.INFO)

SEED = 2018
random.seed(SEED)
np.random.seed(SEED)


def get_query_doc_ids(filepath):
  query_ids, doc_ids = set(), set()
  samples = load_train_test_file(filepath)
  query_ids |= set([s[0] for s in samples])
  doc_ids |= set([s[1] for s in samples])
  if filepath.endswith('pairwise'):
    doc_ids |= set(([s[2] for s in samples]))
  return query_ids, doc_ids


def filter_samples(filepath_old, filepath_new, filter_query, filter_doc):
  samples = load_train_test_file(filepath_old)
  if filepath_old.endswith('pointwise'):
    samples = [s for s in samples if s[0] not in filter_query and s[1] not in filter_doc]
  elif filepath_old.endswith('pairwise'):
    samples = [s for s in samples if s[0] not in filter_query and s[1] not in filter_doc and
           s[2] not in filter_doc]
  save_train_test_file(samples, filepath_new)


def generate_train_test():
  data_dir = args.data_dir
  query_filepath = os.path.join(data_dir, 'query')
  judge_filepath = os.path.join(data_dir, 'judgement')
  run_filepath = os.path.join(data_dir, 'run')
  # split train and test dataset based on queries rather than qid
  query_dict = load_from_query_file(query_filepath)
  unique_queries = np.unique(list(query_dict.values()))
  np.random.shuffle(unique_queries)
  train_size = int(len(unique_queries) * args.train_test_ratio)
  test_size = len(unique_queries) - train_size
  if train_size <= 0 or test_size <= 0:
    raise Exception('train test dataset size is incorrect')
  print('#unique queries: {}, train size: {}, test size: {}'
      .format(len(unique_queries), train_size, test_size))
  train_queries = set(unique_queries[:train_size])
  test_queries = set(unique_queries[train_size:])    
  train_qids = set([q for q in query_dict if query_dict[q] in train_queries])
  test_qids = set([q for q in query_dict if query_dict[q] in test_queries])
  miss_docs = set()
  have_docs = set()
  train_samples = []
  test_samples = []
  qd_judge = load_judge_file(judge_filepath)
  for q in qd_judge:
    for d in qd_judge[q]:
      if qd_judge[q][d] is None: # skip documents without judgement
        continue
      if not os.path.exists(os.path.join(data_dir, 'docs', d + '.html')):
        miss_docs.add(d)
        continue
      have_docs.add(d)
      if q in train_qids:
        train_samples.append((q, d, qd_judge[q][d]))
      elif q in test_qids and not os.path.exists(run_filepath):
        test_samples.append((q, d, qd_judge[q][d]))
  if os.path.exists(run_filepath):
    run_result = load_run_file(run_filepath)
    for q, _, d, rank, score, _ in run_result:
      if qd_judge[q][d] is None: # skip documents without judgement
        continue
      if not os.path.exists(os.path.join(data_dir, 'docs', d + '.html')):
        miss_docs.add(d)
        continue
      have_docs.add(d)
      if q in test_qids:
        test_samples.append((q, d, qd_judge[q][d]))
  print('have {} docs, miss {} docs'.format(len(have_docs), len(miss_docs)))
  save_train_test_file(train_samples, os.path.join(data_dir, 'train.pointwise'))
  save_train_test_file(test_samples, os.path.join(data_dir, 'test.pointwise'))


def load_from_html_cascade(filename, binary=False, field=['title', 'body']):
  try:
    result = load_from_html(filename, binary=binary, field=field)
  except jpype.JException(java.lang.StackOverflowError) as e:
    logging.warn('boilerpipe exception: {}'.format(filename))
    result = load_from_html(filename, use_boilerpipe=False, binary=binary, field=field)
  return result


def preprocess(field='body'):
  load_boilerpipe()
  binary = args.binary_html
  data_dir = args.data_dir
  max_vocab_size = args.max_vocab_size
  docs_dir = os.path.join(data_dir, 'docs')
  query_filepath = os.path.join(data_dir, 'query')
  train_filepath = os.path.join(data_dir, 'train.pointwise')
  test_filepath = os.path.join(data_dir, 'test.pointwise')
  vocab = Vocab(max_size=max_vocab_size)
  train_query_ids, train_doc_ids = get_query_doc_ids(train_filepath)
  test_query_ids, test_doc_ids = get_query_doc_ids(test_filepath)
  query_ids = train_query_ids | test_query_ids
  doc_ids = train_doc_ids | test_doc_ids
  print('total query: {}, total doc: {}'.format(len(query_ids), len(doc_ids)))
  query_dict = load_from_query_file(query_filepath)
  doc_dict = {}
  for qid in sorted(train_query_ids):
    for term in query_dict[qid].split():
      vocab.add(term)
  count = 0
  for docid in sorted(train_doc_ids):
    count += 1
    if count % 10000 == 0:
      print('processed {}w docs'.format(count // 10000))
    loaded_html = load_from_html_cascade(os.path.join(docs_dir, docid + '.html'), binary=binary, field=[field])
    doc_dict[docid] = loaded_html[field]
    #print(docid)
    #print(' '.join(doc_dict[docid]))
    #input()
    for term in doc_dict[docid]:
      vocab.add(term)
  vocab.build()
  vocab.save_to_file(os.path.join(data_dir, 'vocab'))
  empty_qid, empty_docid = set(), set()
  with open(os.path.join(data_dir, 'query.prep'), 'w') as fp:
    for qid in sorted(query_ids):
      qt = query_dict[qid].split()
      if len(qt) == 0:
        empty_qid.add(qid)
        continue
      fp.write('{}\t{}\n'.format(qid, ' '.join(map(lambda x: str(x), vocab.encode(qt)))))
  with open(os.path.join(data_dir, 'docs.prep'), 'w') as fp:
    for docid in sorted(doc_ids):
      if docid in doc_dict:
        doc_text = doc_dict[docid]
      else:
        doc_text = load_from_html_cascade(os.path.join(docs_dir, docid + '.html'), binary=binary, field=[field])[field]
      if len(doc_text) == 0:
        empty_docid.add(docid)
        continue
      fp.write('{}\t{}\n'.format(docid, ' '.join(map(lambda x: str(x), vocab.encode(doc_text)))))
  print('have {} empty query, have {} empty doc'.format(len(empty_qid), len(empty_docid)))
  filter_samples(train_filepath, '{}.prep.{}'.format(*train_filepath.rsplit('.', 1)), empty_qid, empty_docid)
  filter_samples(test_filepath, '{}.prep.{}'.format(*test_filepath.rsplit('.', 1)), empty_qid, empty_docid)


def word_vector_transform():
  print('loading word vector ...')
  wv = WordVector(filepath=args.word_vector_path, first_line=args.first_line)
  vocab = Vocab(filepath=os.path.join(args.data_dir, 'vocab'), file_format=args.format)
  print('transforming ...')
  wv.transform(vocab.get_word_list(), oov_filepath=os.path.join(args.data_dir, 'oov.txt'), 
    oov_at_end=False) # don't use oov_at_end because it is problematic
  print('saving ...')
  wv.save_to_file(os.path.join(args.data_dir, 'w2v'))


def prep_query_log():
  query_log_filepath = args.query_log
  data_dir = args.data_dir
  query_dict = {}
  doc_dict = {}
  query_count = defaultdict(lambda: 0)
  qd_ctr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
  ind = 0
  last_query = '#'
  for fetch in load_query_log(query_log_filepath, format='bing', iterable=True):
    ind += 1
    if ind % 1000000 == 0:
      print('processed {} m'.format(ind / 1000000))
    uid, sid, sstime, nq, qid, q, qtime, nc, rank, url, ctime, dtime = fetch
    if q not in query_dict:
      query_dict[q] = len(query_dict)
    if url not in doc_dict:
      doc_dict[url] = len(doc_dict)
    qd_ctr[query_dict[q]][doc_dict[url]][1] += 1
    qd_ctr[query_dict[q]][doc_dict[url]][0] += int(dtime != 0)
    if uid + qid + qtime != last_query:
      query_count[q] += 1
      last_query = uid + qid + qtime
  for qid in qd_ctr:
    for docid in qd_ctr[qid]:
      qd_ctr[qid][docid] = qd_ctr[qid][docid][0] / qd_ctr[qid][docid][1]
  save_query_file([(v, k) for k, v in query_dict.items()], os.path.join(data_dir, 'query_all'))
  save_query_file(sorted(query_count.items(), key=lambda x: -x[1]), os.path.join(data_dir, 'query_freq'))
  save_judge_file(qd_ctr, os.path.join(data_dir, 'judgement_DCTR'))
  save_query_file([(v, k) for k, v in doc_dict.items()], os.path.join(data_dir, 'docid_to_url'))


def shuqi_bing_redirect():
  MARK = b'\t-----\t'
  data_dir = args.data_dir
  shuqi_bing_web_dir = args.shuqi_bing_web_dir
  docid_to_url = load_from_query_file(os.path.join(data_dir, 'docid_to_url'))
  print('#all url: {}'.format(len(docid_to_url)))
  url_to_docid = {v: k for k, v in docid_to_url.items()}
  count = 0
  wrong_url_count = 0
  with open(os.path.join(shuqi_bing_web_dir, 'allweb.txt'), 'r') as fp:
    for l in fp:
      l = l.strip()
      url, ind = l.split('\t')
      if url not in url_to_docid:
        wrong_url_count += 1
        continue
      old_path = os.path.join(shuqi_bing_web_dir, 'web{}.txt'.format(ind))
      if not os.path.exists(old_path):
        continue
      count += 1
      if count % 100000 == 0:
        print('count: {}w'.format(count / 10000))
      new_ind = url_to_docid[url]
      with open(os.path.join(data_dir, 'docs', new_ind + '.html'), 'wb') as nh:
        try:
          h = open(old_path, 'rb').read()
        except:
          print('read error: {}'.format(old_path))
          raise
        nh.write(h[h.find(MARK) + len(MARK):])
  print('#downloaded url: {}, #wrong url: {}'.format(count, wrong_url_count))


def filter_query():
  data_dir = args.data_dir
  min_query_freq = args.min_query_freq
  query2freq = load_from_query_file(os.path.join(data_dir, 'query_freq'))
  qid2query = load_from_query_file(os.path.join(data_dir, 'query_all'))
  save_query_file([(k, v) for k, v in qid2query.items() if int(query2freq[v]) >= min_query_freq], 
          os.path.join(data_dir, 'query'))


def click_model_to_rel(do=['train', 'test'], files=['train.prep.pointwise', 'test.prep.pointwise']):
  train_file = os.path.join(args.data_dir, files[0])
  test_file = os.path.join(args.data_dir, files[1])
  train_click = load_judge_file(train_file, scale=float)
  clicks = []
  for q in train_click:
    for d in train_click[q]:
      clicks.append(train_click[q][d])
  clicks = sorted(clicks)
  ratio = [(0, 0.8), (1, 0.95), (2, 1)] # the cumulative distribution of relevance label
  threshold = []
  k = 0
  last = '#'
  for i in range(len(clicks)):
    while i / len(clicks) >= ratio[k][1]:
      k += 1
    if last != '#' and last[0] != ratio[k][0]:
      threshold.append(last)
    last = [ratio[k][0], clicks[i]]
  threshold.append(last)
  print('ratio: {}'.format(ratio))
  print('threshold: {}'.format(threshold))
  #threshold = [[0, 0.05], [1, 0.3], [2, 1]]  # my guess
  def click2rel(click):
    k = 0
    while click > threshold[k][1]:
      k += 1
    return threshold[k][0]
  # save
  def map_fn(sample):
    nonlocal click2rel
    return PointwiseSample(sample.qid, sample.docid, click2rel(sample.label), sample.query, sample.doc)
  if 'train' in do:
    prep_file_mapper(train_file, train_file + '.rel', method='sample', func=float, map_fn=map_fn)
  if 'test' in do:
    prep_file_mapper(test_file, test_file + '.rel', method='sample', func=float, map_fn=map_fn)


def click_to_rel():
  data_dir = args.data_dir
  judge_click = os.path.join(data_dir, 'judgement_DCTR')
  judge_refer = args.judgement_refer
  judge_click = load_judge_file(judge_click, scale=float)
  judge_refer = load_judge_file(judge_refer, scale=int)
  rels = []
  for q in judge_refer:
    for d in judge_refer[q]:
      rels.append(judge_refer[q][d])
  clicks = []
  for q in judge_click:
    for d in judge_click[q]:
      clicks.append(judge_click[q][d])
  rels = sorted(rels)
  clicks = sorted(clicks)
  if len(rels) <= 0 or len(clicks) <= 0:
    raise Exception('judgement has no record')
  ratio = []
  last = '#'
  for i in range(len(rels)):
    r = rels[i]
    if r != last:
      ratio.append([r, 0])
      if len(ratio) > 1:
        ratio[-2][1] = i / len(rels)
    last = r
  ratio[-1][1] = 1
  threshold = []
  k = 0
  last = '#'
  for i in range(len(clicks)):
    while i / len(clicks) >= ratio[k][1]:
      k += 1
    if last != '#' and last[0] != ratio[k][0]:
      threshold.append(last)
    last = [ratio[k][0], clicks[i]]
  threshold.append(last)
  print('ratio: {}'.format(ratio))
  print('threshold: {}'.format(threshold))
  threshold = [[0, 0.05], [1, 0.3], [2, 1]] # my guess
  judge_rel = defaultdict(lambda: defaultdict(lambda: None))
  def click2rel(click):
    k = 0
    while click > threshold[k][1]:
      k += 1
    return threshold[k][0]
  for q in judge_click:
    for d in judge_click[q]:
      judge_rel[q][d] = click2rel(judge_click[q][d])
  save_judge_file(judge_rel, os.path.join(data_dir, 'judgement_rel'))


def filter_judgement():
  filtered_ext = ['.pdf', '.ppt', '.pptx', '.doc', '.docx', '.txt']
  filtered_ext = tuple(filtered_ext + [ext.upper() for ext in filtered_ext])
  allowed_ext = tuple(['html', 'htm', 'com', 'cn', 'asp', 'shtml', 'php'])
  data_dir = args.data_dir
  docid_to_url = load_from_query_file(os.path.join(data_dir, 'docid_to_url'))
  qd_judge = load_judge_file(os.path.join(data_dir, 'judgement_rel'))
  qd_judge_new = defaultdict(lambda: defaultdict(lambda: None))
  count = 0
  for q in qd_judge:
    for d in qd_judge[q]:
      if docid_to_url[d].endswith(filtered_ext):
        count += 1
        continue
      qd_judge_new[q][d] = qd_judge[q][d]
  print('#non-html url: {}'.format(count))
  save_judge_file(qd_judge_new, os.path.join(data_dir, 'judgement'))


def find_gzip():
  data_dir = args.data_dir
  gzip_files = []
  count = 0
  with open(os.path.join(data_dir, 'gzip_files'), 'w') as fp:
    for root, dirs, files in os.walk(os.path.join(data_dir, 'docs')):
      for filename in files:
        if open(os.path.join(root, filename), 'rb').read(2) == b'\x1f\x8b':
          gzip_files.append(filename)
          fp.write('{}\n'.format(filename))
        count += 1
        if count % 10000 == 0:
          print('count: {}w, gzip: {}'.format(count // 10000, len(gzip_files)))
      break
    print('count: {}w, gzip: {}'.format(count // 10000, len(gzip_files)))


def ungzip():
  data_dir = args.data_dir
  gzip_files = args.gzip_files
  count, error_count = 0, 0
  with open(gzip_files, 'r') as fp:
    for l in fp:
      count += 1
      if count % 1000 == 0:
        print(count)
      fn = l.strip().split(':')[0]
      comp_filepath = os.path.join(data_dir, fn)
      '''
      with gzip.open(comp_filepath, 'rb') as f_in:
        with open(comp_filepath + '.ungzip', 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)
      os.remove(comp_filepath)
      shutil.move(comp_filepath + '.ungzip', comp_filepath)
      '''
      try:
        with gzip.GzipFile(comp_filepath, 'rb') as f_in:
          s = f_in.read()
        with open(comp_filepath, 'wb') as f_out:
          f_out.write(s)
      except zlib.error:
        error_count += 1
        print('zlib.error: {}'.format(comp_filepath))
      except:
        error_count += 1
        print('error: {}'.format(comp_filepath))
  print('total: {}, error: {}'.format(count, error_count))


def handle_windows():
  data_dir = args.data_dir
  gzip_files = args.gzip_files
  count= 0
  with open(gzip_files, 'r') as fp:
    for l in fp:
      count += 1
      fn = l.strip().split(':')[0]
      comp_filepath = os.path.join(data_dir, fn)
      with open(comp_filepath, 'rb') as f_in:
        s = f_in.read().replace(b'\r\n', b'\n')
      with open(comp_filepath, 'wb') as f_out:
        f_out.write(s)
  print('total: {}'.format(count))


def gen_pairwise_stream(score_diff_thres=0):
  train_pointwise = os.path.join(args.data_dir, 'train.prep.pointwise')
  test_pointwise = os.path.join(args.data_dir, 'test.prep.pointwise')
  for fn in [train_pointwise, test_pointwise]:
    fn_out = fn.rsplit('.', 1)[0] + '.pairwise'
    pointwise_count = 0
    pairwise_count = 0
    with open(fn_out, 'w') as fout:
      for samples in prep_file_iterator(fn, method='query', func=float, parse=False):
        for i in range(len(samples)):
          for j in range(i+1, len(samples)):
            if samples[i].label >= samples[j].label:
              s1 = samples[i]
              s2 = samples[j]
            else:
              s1 = samples[j]
              s2 = samples[i]
            if s1.qid != s2.qid:
              print('!')
              print(s1, s2)
              input()
            if s1.label - s2.label > score_diff_thres:
              pairwise_count += 1
              fout.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                s1.qid, s1.docid, s2.docid, s1.label - s2.label, s1.query, s1.doc, s2.doc))
        pointwise_count += len(samples)
    print('from {} samples to {} samples in {}'.format(pointwise_count, pairwise_count, fn_out))


def gen_pairwise():
  train_pointwise = os.path.join(args.data_dir, 'train.prep.pointwise')
  test_pointwise = os.path.join(args.data_dir, 'test.prep.pointwise')
  for fn in [train_pointwise, test_pointwise]:
    fn_out = fn.rsplit('.', 1)[0] + '.pairwise'
    samples = load_train_test_file(fn, file_format=args.format, reverse=args.reverse)
    samples_gb_q = groupby(samples, lambda x: x[0])
    with open(fn_out, 'w') as fout:
      all_pointwise = 0
      all_pairwise = 0
      for q, q_samples in samples_gb_q:
        q_samples = list(q_samples)
        count = 0
        for s1 in q_samples:
          for s2 in q_samples:
            if s1[2] > s2[2]:
              count += 1
              fout.write('{}\t{}\t{}\t{}\n'.format(s1[0], s1[1], s2[1], s1[2]-s2[2]))
        #print('query {}, #pointwise {}, #pairwise {}, ratio {}'.format(
        #  q, len(q_samples), count, count/(len(q_samples) or 1)))
        all_pointwise += len(q_samples)
        all_pairwise += count
      print('from {} samples to {} samples in {}'.format(all_pointwise, all_pairwise, fn_out))


def gen_tfrecord_stream(do=['train', 'test']):
  # number of tfrecord file to save
  num_shards = 1
  print('use {} shards'.format(num_shards))
  def _pick_output_shard():
    return random.randint(0, num_shards-1)
  train_filename = os.path.join(args.data_dir, 'train.prep.{}'.format(args.paradigm))
  test_filename = os.path.join(args.data_dir, 'test.prep.{}'.format(args.paradigm))
  files = []
  if 'train' in do:
    files.append(train_filename)
  if 'test' in do:
    files.append(test_filename)
  for fn in files:
    print('convert "{}" ...'.format(fn))
    output_file = fn + '.tfrecord'
    writers = []
    for i in range(num_shards):
      writers.append(tf.python_io.TFRecordWriter(
        '%s-%03i-of-%03i' % (output_file, i, num_shards)))
    for i, sample in enumerate(prep_file_iterator(fn, method='sample', func=float, parse=True)):
      if i % 100000 == 0:
        print('{}w'.format(i // 10000))
      features = {}
      if args.paradigm == 'pointwise':
        q, d = sample.qid, sample.docid
        qb = q.encode('utf-8')
        db = d.encode('utf-8')
        features['docid'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[db]))
        features['doc'] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample.doc))
        features['doclen'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[len(sample.doc)]))
      elif args.paradigm == 'pairwise':
        q, d1, d2, = sample.qid, sample.docid1, sample.docid2
        qb = q.encode('utf-8')
        d1b = d1.encode('utf-8')
        d2b = d2.encode('utf-8')
        features['docid1'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[d1b]))
        features['docid2'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[d2b]))
        features['doc1'] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample.doc1))
        features['doc2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample.doc2))
        features['doc1len'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[len(sample.doc1)]))
        features['doc2len'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[len(sample.doc2)]))
      features['qid'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[qb]))
      features['query'] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample.query))
      features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[sample.label]))
      features['qlen'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(sample.query)]))
      f = tf.train.Features(feature=features)
      example = tf.train.Example(features=f)
      # randomly choose a shard to save the example
      writers[_pick_output_shard()].write(example.SerializeToString())


def gen_tfrecord(bow=False, segmentation=False):
  # number of tfrecord file to save
  num_shards = 1
  print('use {} shards'.format(num_shards))
  def _pick_output_shard():
    return random.randint(0, num_shards-1)
  print('load text file ...')
  if not bow:
    doc_file = os.path.join(args.data_dir, 'docs.prep')
  else:
    doc_file = os.path.join(args.data_dir, 'docs_bow.prep')
    doc_weight_file = os.path.join(args.data_dir, 'docs_bow_weight.prep')
  if segmentation:
    doc_seg_file = os.path.join(args.data_dir, 'docs_seg.prep')
  if args.format == 'ir':
    if not bow:
      query_file = os.path.join(args.data_dir, 'query.prep')
    else:
      query_file = os.path.join(args.data_dir, 'query_bow.prep')
      query_weight_file = os.path.join(args.data_dir, 'query_bow_weight.prep')
  doc_raw = load_prep_file(doc_file, file_format=args.format)
  if args.format == 'ir':
    query_raw = load_prep_file(query_file, file_format=args.format)
  else:
    query_raw = doc_raw
  if bow:
    doc_raw_weight = load_prep_file(doc_weight_file, file_format=args.format, func=float)
    if args.format == 'ir':
      query_raw_weight = load_prep_file(query_weight_file, file_format=args.format, func=float)
    else:
      query_raw_weight = doc_raw_weight
  if segmentation:
    doc_seg = load_prep_file(doc_seg_file, file_format=args.format)
  train_filename = os.path.join(args.data_dir, 'train.prep.{}'.format(args.paradigm))
  test_filename = os.path.join(args.data_dir, 'test.prep.{}'.format(args.paradigm))
  for fn in [train_filename, test_filename]:
    print('convert "{}" ...'.format(fn))
    output_file = fn + '.tfrecord'
    writers = []
    for i in range(num_shards):
      writers.append(tf.python_io.TFRecordWriter(
        '%s-%03i-of-%03i' % (output_file, i, num_shards)))
    samples = load_train_test_file(fn, file_format=args.format, reverse=args.reverse)
    for i, sample in enumerate(samples):
      if i % 100000 == 0:
        print('{}w'.format(i // 10000))
      features = {}
      if args.paradigm == 'pointwise':
        q, d, r = sample
        qb = q.encode('utf-8')
        db = d.encode('utf-8')
        features['docid'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[db]))
        features['doc'] = tf.train.Feature(int64_list=tf.train.Int64List(value=doc_raw[d]))
        if bow:
          features['doc_weight'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=doc_raw_weight[d]))
        if segmentation:
          features['doc_segmentation'] = tf.train.Feature(int64_list=tf.train.Int64List(value=doc_seg[d]))
        features['doclen'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[len(doc_raw[d])]))
      elif args.paradigm == 'pairwise':
        q, d1, d2, r = sample
        qb = q.encode('utf-8')
        d1b = d1.encode('utf-8')
        d2b = d2.encode('utf-8')
        features['docid1'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[d1b]))
        features['docid2'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[d2b]))
        features['doc1'] = tf.train.Feature(int64_list=tf.train.Int64List(value=doc_raw[d1]))
        features['doc2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=doc_raw[d2]))
        if bow:
          features['doc1_weight'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=doc_raw_weight[d1]))
          features['doc2_weight'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=doc_raw_weight[d2]))
        if segmentation:
          features['doc1_segmentation'] = tf.train.Feature(int64_list=tf.train.Int64List(value=doc_seg[d1]))
          features['doc2_segmentation'] = tf.train.Feature(int64_list=tf.train.Int64List(value=doc_seg[d2]))
        features['doc1len'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[len(doc_raw[d1])]))
        features['doc2len'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[len(doc_raw[d2])]))
      features['qid'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[qb]))
      features['query'] = tf.train.Feature(int64_list=tf.train.Int64List(value=query_raw[q]))
      if bow:
        features['query_weight'] = tf.train.Feature(
          float_list=tf.train.FloatList(value=query_raw_weight[q]))
      features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[r]))
      features['qlen'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(query_raw[q])]))
      f = tf.train.Features(feature=features)
      example = tf.train.Example(features=f)
      # randomly choose a shard to save the example
      writers[_pick_output_shard()].write(example.SerializeToString())


def drop_negative():
  keep = 0.15
  train_filename = os.path.join(args.data_dir, 'train.prep.{}'.format(args.paradigm))
  out_filename = os.path.join(args.data_dir, 'train.prep.neg{}.{}'.format(keep, args.paradigm))
  samples = load_train_test_file(train_filename, file_format=args.format, reverse=args.reverse)
  save_train_test_file(
    [s for s in samples if (s[-1] > 0) or (random.random() <= keep)], out_filename, 
    file_format=args.format)


def gen_tfidf():
  doc_file = os.path.join(args.data_dir, 'docs.prep')
  doc_bow_file= os.path.join(args.data_dir, 'docs_bow.prep')
  doc_weight_file = os.path.join(args.data_dir, 'docs_bow_weight.prep')
  doc_raw = load_prep_file_aslist(doc_file, file_format=args.format)
  if args.format == 'ir':
    query_file = os.path.join(args.data_dir, 'query.prep')
    query_bow_file = os.path.join(args.data_dir, 'query_bow.prep')
    query_weight_file = os.path.join(args.data_dir, 'query_bow_weight.prep')
    query_raw = load_prep_file_aslist(query_file, file_format=args.format)
  corpus = [d[1] for d in doc_raw]
  if args.format == 'ir':
    corpus += [q[1] for q in query_raw]
  print('corpus size is: {}, start calculating tfidf vectors ...'.format(len(corpus)))
  vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', tokenizer=lambda x: x.split(), 
    binary=False, min_df=1, norm='l2', sublinear_tf=True, use_idf=True)
  tfidf_vec = vectorizer.fit_transform(corpus)
  doc_vec = tfidf_vec[:len(doc_raw)]
  if args.format == 'ir':
    query_vec = tfidf_vec[len(doc_raw):]
  features = np.array(vectorizer.get_feature_names())
  print('total vocab size: {}, start saving ...'.format(len(features)))
  save_prep_file(doc_bow_file, [(d[0], features[doc_vec[i].indices]) 
    for i,d in enumerate(doc_raw)])
  save_prep_file(doc_weight_file, [(d[0], doc_vec[i].data) 
    for i,d in enumerate(doc_raw)])
  if args.format == 'ir':
    save_prep_file(query_bow_file, [(q[0], features[query_vec[i].indices]) 
      for i,q in enumerate(query_raw)])
    save_prep_file(query_weight_file, [(q[0], query_vec[i].data) 
      for i,q in enumerate(query_raw)])


def multi_qc_xml_field_line_map(arg):
  xml_file, field, map_fn, ind, start, end = arg
  qc_xml_field_line_map(xml_file, field, map_fn, ind, start, end)


def xml_map_fn(text):
  words = []
  for w in word_segment(text):
    w = w.strip()
    if w.find(' ') != -1:
      print(w)
      #input()
    if len(w) > 0:
      words.append(w)
  return ' '.join(words)


def xml_word_seg():
  xml_file = os.path.join(args.data_dir, 'qd.xml') # xml file in sogou-qcl format
  if args.line_count is not None:
    num_lines = args.line_count
  else:
    num_lines = sum(1 for l in open(xml_file, 'r'))
  print('num lines {}'.format(num_lines))
  batch_size = int(np.ceil(num_lines / args.num_thread))
  # controllable pool
  original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
  pool = multiprocessing.Pool(args.num_thread)
  signal.signal(signal.SIGINT, original_sigint_handler)
  try:
    print('starting {} jobs'.format(args.num_thread))
    res = pool.map_async(multi_qc_xml_field_line_map, [(xml_file, ('<query>', '<title>', '<content>'), xml_map_fn,
        i, i * batch_size, i * batch_size + batch_size) for i in range(args.num_thread)])
    print('waiting for results')
    res.get(9999999)  # without the timeout this blocking call ignores all signals.
  except KeyboardInterrupt:
    print('caught KeyboardInterrupt, terminating workers')
    pool.terminate()
  else:
    print('normal termination')
    pool.close()
  print('start joining')
  pool.join()


def xml_field_maping(field):
  if field == 'body':
    field_in_xml = 'content'
  elif field == 'title':
    field_in_xml = 'title'
  return field_in_xml


def xml_train_test_prep(field='body', relevance='TACM'):
  train_file = os.path.join(args.data_dir, 'qd.xml.seg.train')
  test_file = os.path.join(args.data_dir, 'qd.xml.seg.test')
  max_vocab_size = args.max_vocab_size
  train_word_file = os.path.join(args.data_dir, 'train.pointwise')
  test_word_file = os.path.join(args.data_dir, 'test.pointwise')
  train_prep_file = os.path.join(args.data_dir, 'train.prep.pointwise')
  test_prep_file = os.path.join(args.data_dir, 'test.prep.pointwise')
  vocab_file = os.path.join(args.data_dir, 'vocab')
  field_in_xml = xml_field_maping(field)
  print('build vocab ...')
  vocab = Vocab(max_size=max_vocab_size)
  for i, qd in enumerate(qd_xml_iterator(train_file)):
    '''
    query = qd.find('./query').text
    words = query.split(' ')
    for doc in qd.findall('./doc/{}'.format(field_in_xml)):
      words.extend(doc.text.split(' '))
    '''
    if i % 10000 == 0:
      print('{}w'.format(i//10000))
    query = qd['query']
    words = query.split(' ')
    for doc in qd['doc']:
      words.extend(doc[field_in_xml].split(' '))
    for w in words:
      vocab.add(w)
  vocab.build()
  vocab.save_to_file(vocab_file)
  for from_file, word_file, prep_file in \
    [(train_file, train_word_file, train_prep_file), (test_file, test_word_file, test_prep_file)]:
    qd_xml_to_prep(from_file, prep_file, vocab, field_in_xml=field_in_xml, relevance=relevance)


def xml_prep(from_file, prep_file, field='body', relevance='TACM'):
  field_in_xml = xml_field_maping(field)
  vocab = Vocab(filepath=os.path.join(args.data_dir, 'vocab'), file_format=args.format)
  qd_xml_to_prep(from_file, prep_file, vocab, field_in_xml=field_in_xml, relevance=relevance)


def prep_to_entityduet_format():
  train_file = os.path.join(args.data_dir, 'train.prep.pairwise')
  dev_file = os.path.join(args.data_dir, 'test.prep.pointwise')
  test_file = os.path.join(args.data_dir, 'test.prep.pointwise')
  vocab_file = os.path.join(args.data_dir, 'vocab')
  emb_file = os.path.join(args.data_dir, 'w2v')
  train_file_out = os.path.join(args.out_dir, 'train_pair.pkl')
  dev_file_out = os.path.join(args.out_dir, 'dev.pkl')
  test_file_out = os.path.join(args.out_dir, 'test.pkl')
  vocab_file_out = os.path.join(args.out_dir, 'vocab.txt')
  emb_file_out = os.path.join(args.out_dir, 'embed.txt')
  def id_map_fn(ids):
    return [id + 1 for id in ids]
  def label_map_fn(label):
    if label > 0:
      return 1
    return 0
  # save train, dev, test data
  for in_file, out_file in [(train_file, train_file_out), (dev_file, dev_file_out), (test_file, test_file_out)]:
    transformed_data = []
    print('transforming {} ...'.format(in_file))
    if in_file.endswith('pointwise'):
      mode = 1
      func = int
    elif in_file.endswith('pairwise'):
      mode = 2
      func = float
    for sample in prep_file_iterator(in_file, method='sample', func=func, parse=True):
      if mode == 1:
        transformed_data.append(
          (id_map_fn(sample.query), id_map_fn(sample.doc), label_map_fn(sample.label), sample.qid))
      elif mode == 2:
        transformed_data.append(
          (id_map_fn(sample.query), id_map_fn(sample.doc1), id_map_fn(sample.doc2)))
    print('saving to {}'.format(out_file))
    with open(out_file, 'wb') as fout:
      pickle.dump(transformed_data, fout, protocol=2)
  # save vocab
  print('saving to {}'.format(vocab_file_out))
  vocab = Vocab(filepath=vocab_file, file_format=args.format)
  words = ['<PAD>'] + vocab.get_word_list()
  with open(vocab_file_out, 'w') as fout:
    fout.write('\n'.join(words) + '\n')
  # save emb
  print('saving to {}'.format(emb_file_out))
  wv = WordVector(filepath=emb_file, first_line=args.first_line)
  vector = np.concatenate([np.zeros_like(wv.vectors[:1]), wv.vectors], axis=0)
  vector.dump(emb_file_out)


def chkpt_rename(checkpoint_dir, new_checkpoint_dir, dry_run=True, map_name=dict()):
  with tf.Session() as sess:
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
      # Load the variable
      var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
      # Set the new name
      new_name = var_name
      if var_name in map_name:
        new_name = map_name[var_name]
        print('%s would be renamed to %s.' % (var_name, new_name))
      var = tf.Variable(var, name=new_name)
    if not dry_run:
      # Save the variables
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initializer())
      saver.save(sess, new_checkpoint_dir)


if __name__ == '__main__':
  if args.action == 'prep':
    '''
    usage: python prep.py -a prep -d data_dir -m 50000 --binary_html
    remember to set the field to be extracted.
    '''
    preprocess(field='title')
  elif args.action == 'gen':
    generate_train_test()
  elif args.action == 'w2v':
    '''
    usage: python prep.py -a w2v -d data_dir -f text --word_vector_path word_vec_path --first_line
    remember to remove first_line when needed.
    don't use oov_at_end.
    '''
    word_vector_transform()
  elif args.action == 'prep_query_log':
    prep_query_log()
  elif args.action == 'filter_query':
    filter_query()
  elif args.action == 'shuqi_bing_redirect':
    shuqi_bing_redirect()
  elif args.action == 'click_to_rel':
    click_to_rel()
  elif args.action == 'click_model_to_rel':
    '''
    usage: python prep.py -a click_model_to_rel -d data_dir
    remember to modify do and files.
    '''
    click_model_to_rel(do=['train', 'test'], files=['train.prep.pointwise', 'test.prep.pointwise'])
  elif args.action == 'filter_judgement':
    filter_judgement()
  elif args.action == 'find_gzip':
    find_gzip()
  elif args.action == 'ungzip':
    ungzip()
  elif args.action == 'handle_windows':
    handle_windows()
  elif args.action == 'gen_pairwise':
    '''
    usage: python prep.py -a gen_pairwise -d data_dir [-f text --reverse --use_stream]
    remember to set score_diff_thres.
    '''
    if args.use_stream:
      gen_pairwise_stream(score_diff_thres=0.1)
    else:
      gen_pairwise()
  elif args.action == 'gen_tfrecord':
    '''
    usage: python prep.py -a gen_tfrecord -d data_dir -f text -p pointwise [--use_stream]
    remember to modify the do, bow, and segmentation parameter.
    '''
    if args.use_stream:
      gen_tfrecord_stream(do=['train', 'test'])
    else:
      gen_tfrecord(bow=False, segmentation=True)
  elif args.action == 'drop_negative':
    # random drop some negative samples to make training balanced
    drop_negative()
  elif args.action == 'gen_tfidf':
    # generate tfidf-like dataset in which each document is a bow with tfidf as weights.
    gen_tfidf()
  elif args.action == 'xml_word_seg':
    '''
    usage: python prep.py -a xml_word_seg -d data_dir --line_count line_count -t 4
    remember that the line_count given by `wc -l` might be different that give by python.
    '''
    xml_word_seg()
  elif args.action == 'xml_train_test_prep':
    '''
    usage: python prep.py -a xml_train_test_prep -d data_dir --max_vocab_size 100000
    remember to modify the field and relevance.
    '''
    xml_train_test_prep(field='body', relevance='TACM')
  elif args.action == 'xml_prep':
    '''
    usage: python prep.py -a xml_prep -d data_dir --format ir
    remember to modify the filepath, field, and relevance.
    '''
    xml_prep(from_file='data/sogou_qcl_08/qd.xml.seg.test.filter',
      prep_file='data/sogou_qcl_08/test.prep.pointwise.filter', field='title', relevance='TACM')
  elif args.action == 'prep_to_entityduet_format':
    '''
    usage: python prep.py -a prep_to_entityduet_format -d data_dir -o out_dir --format ir [--first_line]
    '''
    prep_to_entityduet_format()
  elif args.action == 'chkpt_rename':
    '''
    usage: python prep.py -a chkpt_rename
    remember to modify the dir and map_name dict.
    '''
    chkpt_rename(checkpoint_dir='', new_checkpoint_dir='', dry_run=False, map_name=dict())
  elif args.action == 'xml_filter':
    filepath = 'data/sogou_qcl_08/qd.xml.seg.test'
    out_filepath = filepath + '.filter'
    def filter_fn(qd):
      if int(qd['query_frequency']) < 100:
        return True
      return False
    qd_xml_filter(filepath, out_filepath, filter_fn=filter_fn)
  elif args.action == 'recover':
    cache_line = []
    with open('data/sogou_qcl_08/qd.xml.seg', 'r') as fin, \
      open('data/sogou_qcl_08/qd.xml.seg.recover', 'w') as fout:
      for i, l in enumerate(fin):
        if not l.lstrip().startswith('<'):
          print(i)
          cache_line[-1] = cache_line[-1].rstrip('\n') + ' '
        cache_line.append(l)
        if len(cache_line) >= 2:
          fout.write(cache_line[0])
          cache_line = cache_line[1:]
      for l in cache_line:
        fout.write(l)
  else:
    raise NotImplementedError('action not supported')
