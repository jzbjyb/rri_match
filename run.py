import argparse, logging, os, random
from itertools import groupby
import numpy as np
from utils import Vocab, WordVector, load_prep_file, load_train_test_file
from rri import RRI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run')
    parser.add_argument('-a', '--action', help='action', type=str, default='train_test')
    parser.add_argument('-D', '--debug', help='whether to use debug log level', action='store_true')
    parser.add_argument('-d', '--data_dir', help='data directory', type=str)
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
            for i in range(len(X)):
                q_x = X[i]
                q_y = y[i]
                n_sample = len(q_x['query'])
                if n_sample <= 0:
                    continue
                for j in range(0, n_sample, batch_size):
                    yield {
                        'query': data_pad(q_x['query'][j:j+batch_size], q_x['max_q_len'], np.int32),
                        'doc': data_pad(q_x['doc'][j:j+batch_size], q_x['max_d_len'], np.int32),
                        'qd_size': np.array(q_x['qd_size'][j:j+batch_size], dtype=np.int32),
                        'relevance': np.array(q_y['relevance'][j:j+batch_size], dtype=np.int32),
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
    rri = RRI(max_q_len=max_q_len, max_d_len=max_d_len, max_jump_step=10, word_vector=wv.vectors,
              interaction='dot', glimpse='fix_hard', glimpse_fix_size=10, jump='max_hard', represent='sum_hard',
              rel_level=5, learning_rate=0.1, random_seed=SEED, n_epochs=10, batch_size=100, batcher=batcher,
              verbose=True, save_epochs=None, reuse_model=None, save_model=None)
    list(rri.fit_iterable(train_X, train_y))


if __name__ == '__main__':
    if args.action == 'train_test':
        train_test()