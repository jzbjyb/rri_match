import os, argparse, logging, jpype
import numpy as np
from jpype import *
from utils import Vocab, WordVector, load_from_html, load_from_query_file, load_train_test_file, save_train_test_file
from config import CONFIG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('-a', '--action', help='action', type=str, default='prep')
    parser.add_argument('-m', '--max_vocab_size', help='max vocabulary size', type=int, default=50000)
    parser.add_argument('-d', '--data_dir', help='data directory', type=str)
    parser.add_argument('-r', '--train_test_ratio', help='the ratio of train and test dataset',
                        type=float, default=0.8)
    parser.add_argument('-w', '--word_vector_path', help='the filepath of word vector', type=str)
    parser.add_argument('-D', '--debug', help='whether to use debug log level', action='store_true')
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


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
    query_dict = load_from_query_file(query_filepath)
    qids = list(query_dict.keys())
    np.random.shuffle(qids)
    train_size = int(len(qids) * args.train_test_ratio)
    test_size = len(qids) - train_size
    if train_size <= 0 or test_size <= 0:
        raise Exception('train test dataset size is incorrect')
    logging.info('train size: {}, test size: {}'.format(train_size, test_size))
    train_qids = set(qids[:train_size])
    test_qids = set(qids[train_size:])
    miss_docs = set()
    have_docs = set()
    train_samples = []
    test_samples = []
    with open(judge_filepath, 'r') as judge_fp:
        for l in judge_fp:
            q, d, r = l.rstrip().split('\t')
            r = int(r)
            if not os.path.exists(os.path.join(data_dir, 'docs', d + '.html')):
                miss_docs.add(d)
                continue
            have_docs.add(d)
            if q in train_qids:
                train_samples.append((q, d, r))
            elif q in test_qids:
                test_samples.append((q, d, r))
    logging.info('have {} docs, miss {} docs'.format(len(have_docs), len(miss_docs)))
    save_train_test_file(train_samples, os.path.join(data_dir, 'train.pointwise'))
    save_train_test_file(test_samples, os.path.join(data_dir, 'test.pointwise'))


def load_from_html_cascade(filename):
    try:
        result = load_from_html(filename)
    except jpype.JException(java.lang.StackOverflowError) as e:
        logging.info('boilerpipe exception: {}'.format(filename))
        result = load_from_html(filename, use_boilerpipe=False)
    return result


def preprocess():
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
    query_dict = load_from_query_file(query_filepath)
    doc_dict = {}
    for qid in sorted(train_query_ids):
        for term in query_dict[qid].split():
            vocab.add(term)
    count = 0
    for docid in sorted(train_doc_ids):
        count += 1
        #if count % 1000 == 0:
        #    logging.info('processed {} docs'.format(count))
        doc_body = load_from_html_cascade(os.path.join(docs_dir, docid + '.html'))['body']
        doc_dict[docid] = doc_body
        #print(docid)
        #print(' '.join(doc_body))
        #input()
        for term in doc_body:
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
                doc_body = doc_dict[docid]
            else:
                doc_body = load_from_html_cascade(os.path.join(docs_dir, docid + '.html'))['body']
            if len(doc_body) == 0:
                empty_docid.add(docid)
                continue
            fp.write('{}\t{}\n'.format(docid, ' '.join(map(lambda x: str(x), vocab.encode(doc_body)))))
    logging.info('have {} empty query, have {} empty doc'.format(len(empty_qid), len(empty_docid)))
    filter_samples(train_filepath, '{}.prep.{}'.format(*train_filepath.rsplit('.', 1)), empty_qid, empty_docid)
    filter_samples(test_filepath, '{}.prep.{}'.format(*test_filepath.rsplit('.', 1)), empty_qid, empty_docid)


def word_vector_transform():
    print('loading word vector ...')
    wv = WordVector(filepath=args.word_vector_path)
    vocab = Vocab(filepath=os.path.join(args.data_dir, 'vocab'))
    print('transforming ...')
    wv.transform(vocab.get_word_list())
    print('saving ...')
    wv.save_to_file(os.path.join(args.data_dir, 'w2v'))


if __name__ == '__main__':
    if args.action == 'prep':
        preprocess()
    elif args.action == 'gen':
        generate_train_test()
    elif args.action == 'w2v':
        word_vector_transform()
