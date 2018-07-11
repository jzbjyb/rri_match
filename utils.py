import contextlib, sys, re, logging, time, html
from collections import defaultdict
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup, UnicodeDammit
from boilerpipe.extract import Extractor
from nltk.tokenize import word_tokenize


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def tf_jacobian(y_flat, x):
    n = y_flat.shape[1]
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[:, j], x)[0])),
        loop_vars)
    jacobian = jacobian.stack()
    x_len = len(x.get_shape())
    jacobian = tf.transpose(jacobian, [1, 0] + list(range(2, x_len + 1)))
    return jacobian


def load_query_log(filepath, format='bing', iterable=True):
    if format not in {'bing'}:
        raise NotImplementedError('format not supported')
    with open(filepath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if len(l) == 0:
                continue
            if format == 'bing':
                uid, sid, sstime, nq, qid, q, qtime, nc, rank, url, ctime, dtime = l.split('\t')
                rank = int(rank)
                dtime = float(dtime)
                fetch = [uid, sid, sstime, nq, qid, q, qtime, nc, rank, url, ctime, dtime]
            if iterable:
                yield fetch


def load_judge_file(filepath, scale=int):
    qd_judge = defaultdict(lambda: defaultdict(lambda: None))
    with open(filepath, 'r') as fp:
        for l in fp:
            q, d, r = l.rstrip().split('\t')
            r = scale(r)
            qd_judge[q][d] = r
    return qd_judge


def save_judge_file(qd_judge, filepath):
    with open(filepath, 'w') as fp:
        for qid in qd_judge:
            for docid in qd_judge[qid]:
                fp.write('{}\t{}\t{}\n'.format(qid, docid, qd_judge[qid][docid]))


def load_run_file(filepath):
    result = []
    with open(filepath, 'r') as fp:
        for l in fp:
            q, q0, d, rank, score, tag = l.rstrip().split(' ')
            rank = int(rank)
            score = float(score)
            result.append((q, q0, d, rank, score, tag))
    return result


def load_prep_file(filepath):
    result = {}
    with open(filepath, 'r') as fp:
        for l in fp:
            l = l.rstrip('\n')
            if len(l) == 0:
                continue
            k, ws = l.split('\t')
            result[k] = [int(w) for w in ws.split(' ') if len(w) > 0]
    return result


def load_train_test_file(filepath):
    samples = []
    with open(filepath, 'r') as fp:
        for l in fp:
            if filepath.endswith('pointwise'):
                q, d, r = l.split('\t')
                samples.append((q, d, int(r)))
            elif filepath.endswith('pairwise'):
                q, d1, d2, r = l.split('\t')
                samples.append((q, d1, d2, float(r)))
    return samples


def save_train_test_file(samples, filepath):
    with open(filepath, 'w') as fp:
        for s in samples:
            if filepath.endswith('pointwise'):
                fp.write('{}\t{}\t{}\n'.format(s[0], s[1], s[2]))
            elif filepath.endswith('pairwise'):
                fp.write('{}\t{}\t{}\t{}\n'.format(s[0], s[1], s[2], s[3]))


def load_word_vector(filepath, is_binary=False):
    if is_binary:
        raise NotImplementedError()
    words = []
    vectors = []
    with open(filepath, 'r') as fp:
        vocab_size, dim = fp.readline().split()
        vocab_size = int(vocab_size)
        dim = int(dim)
        for i in range(vocab_size):
            rl = fp.readline().rstrip()
            l = rl.split(' ')
            words.append(l[0])
            v = [float(f) for f in l[1:]]
            if len(v) != dim:
                raise Exception('word vector format error')
            vectors.append(v)
    words = np.array(words, dtype=str)
    vectors = np.array(vectors, dtype=np.float32)
    return words, vectors


def clean_text(text):
    text = re.sub('[^a-zA-Z0-9 \n\.-]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text, flags=re.UNICODE).lower()
    return text


def my_word_tokenize(text):
    text = re.sub(r'[+=/]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text, flags=re.UNICODE).lower()
    text = word_tokenize(text) # time consuming
    return text


def load_from_html(filename, use_boilerpipe=True, use_nltk=True, use_regex=True, binary=False):
    if binary:
        charset = UnicodeDammit(open(filename, 'rb').read())
        charset = charset.original_encoding
        try:
            content = open(filename, 'r', encoding=charset).read()
        except Exception as e:
            # if has error, return empty results
            logging.warn('encode error: {}, {}'.format(filename, e))
            return {
                'title': [],
                'body': []
            }
    else:
        content = open(filename, 'r', encoding='utf-8').read()
    start = time.time()
    if not use_regex or not use_boilerpipe:
        bs = BeautifulSoup(content, 'html.parser')
    if use_regex:
        match = re.search(r'<title.*?>(.+?)</title>', content[:5000], re.DOTALL|re.IGNORECASE)
        title = match.group(1) if match else ''
        title = html.unescape(title).strip()
    else:
        if bs.title != None and bs.title.string != None:
            title = bs.title.string.strip()
        else:
            title = ''
    t1 = time.time() - start
    start = time.time()
    if use_boilerpipe:
        extractor = Extractor(extractor='ArticleExtractor', html=content) # time consuming
        body = extractor.getText()
    else:
        body = bs.select('body')
        if len(body) <= 0:
            body = bs
        else:
            body = body[0]
        # remove all useless label
        [x.extract() for x in body.findAll('script')]
        [x.extract() for x in body.findAll('style')]
        [x.extract() for x in body.findAll('meta')]
        [x.extract() for x in body.findAll('link')]
        body = body.text
    t2 = time.time() - start
    start = time.time()
    result = {
        'title': my_word_tokenize(title) if use_nltk else clean_text(title).split(' '),
        'body': my_word_tokenize(body) if use_nltk else clean_text(body).split(' '),
    }
    t3 = time.time() - start
    #print('{}\t{}\t{}'.format(t1, t2, t3))
    return result


def load_from_query_file(filepath):
    query_dict = {}
    with open(filepath, 'r') as fp:
        for l in fp:
            qid, query = l.rstrip('\n').split('\t')
            query_dict[qid] = query
    return query_dict


def save_query_file(queries, filepath):
    with open(filepath, 'w') as fp:
        for q in queries:
            fp.write('{}\t{}\n'.format(q[0], q[1]))


class Vocab(object):
    def __init__(self, max_size=None, filepath=None):
        self.UNK = '<UNK>'
        self.word2count = {}
        self.word2ind = {}
        self.vocab_size = 0
        self.max_size = max_size or sys.maxsize
        if filepath != None:
            self.load_from_file(filepath)


    def get_word_list(self):
        return [self.ind2word[i] for i in range(self.vocab_size)]


    def add(self, word):
        if word not in self.word2count:
            self.word2count[word] = 0
        self.word2count[word] += 1


    def build(self):
        word2count_sorted = sorted(self.word2count.items(), key=lambda x: -x[1])
        for i in range(len(word2count_sorted)):
            if self.max_size != None and i >= self.max_size:
                break
            self.word2ind[word2count_sorted[i][0]] = i + 1 # 0 is for 'UNK'
        self.vocab_size = i + 1
        while self.UNK in self.word2ind:
            self.UNK = '<' + self.UNK + '>'
        self.word2ind[self.UNK] = 0
        self.ind2word = dict(zip(self.word2ind.values(), self.word2ind.keys()))
        logging.info('vocab size: {}, totally: {}'.format(self.vocab_size, len(self.word2count)))


    def encode(self, sequence):
        return [self.word2ind[w] if (w != self.UNK and w in self.word2ind) else self.word2ind[self.UNK]
                for w in sequence]


    def decode(self, sequence):
        return [self.ind2word[i] for i in sequence]


    def save_to_file(self, filepath):
        with open(filepath, 'w') as fp:
            for i in range(self.vocab_size):
                w = self.ind2word[i]
                if w == self.UNK:
                    fp.write('{}\t{}\n'.format(w, 0))
                else:
                    fp.write('{}\t{}\n'.format(w, self.word2count[w]))


    def load_from_file(self, filepath):
        with open(filepath, 'r') as fp:
            ind = 0
            for l in fp:
                w, c = l.split('\t')
                if not ind:
                    self.UNK = w
                else:
                    self.word2count[w] = int(c)
                self.word2ind[w] = ind
                ind += 1
            self.vocab_size = ind
        self.ind2word = dict(zip(self.word2ind.values(), self.word2ind.keys()))


class WordVector(object):
    def __init__(self, filepath=None, is_binary=False, initializer='uniform'):
        if initializer not in {'uniform'}:
            raise Exception('initializer not supported')
        self.initializer = initializer
        if filepath != None:
            self.raw_words, self.raw_vectors = load_word_vector(filepath, is_binary=is_binary)
        self.raw_vocab_size = len(self.raw_words)
        self.raw_words2ind = dict(zip(self.raw_words, range(self.raw_vocab_size)))
        self.dim = self.raw_vectors.shape[1]
        self.vocab_size = self.raw_vectors.shape[0]
        self.words = np.array(self.raw_words)
        self.vectors = np.array(self.raw_vectors)


    def transform(self, new_words, oov_filepath=None):
        start_ind = self.raw_vocab_size
        def new_inder(w):
            nonlocal start_ind
            if w in self.raw_words2ind:
                return self.raw_words2ind[w]
            else:
                start_ind += 1
                return start_ind - 1
        new_ind = [new_inder(w) for w in new_words]
        self.words = np.array(new_words)
        logging.info('total {} words, miss {} words'
                     .format(len(new_words), start_ind - self.raw_vocab_size))
        if oov_filepath != None:
            with open(oov_filepath, 'w') as fp:
                for i in range(len(new_words)):
                    if new_ind[i] >= self.raw_vocab_size:
                        fp.write('{}\n'.format(new_words[i]))
        if self.initializer == 'uniform':
            new_part = np.random.uniform(-.1, .1, [start_ind - self.raw_vocab_size, self.dim])
        self.vectors = np.concatenate([self.raw_vectors, new_part], axis=0)[new_ind]
        self.vocab_size = len(self.words)


    def get_vectors(self, normalize=False):
        if normalize:
            return self.vectors / np.sqrt(np.sum(self.vectors * self.vectors, axis=1, keepdims=True))
        else:
            return self.vectors


    def update(self, new_vectors):
        if new_vectors.shape != self.vectors.shape:
            raise Exception('shape is not correct')
        self.vectors = new_vectors


    def save_to_file(self, filepath, is_binary=False):
        if is_binary:
            raise NotImplementedError()
        with open(filepath, 'w') as fp:
            fp.write('{} {}\n'.format(self.vocab_size, self.dim))
            for i in range(self.vocab_size):
                fp.write('{} {}\n'.format(self.words[i], ' '.join(map(lambda x: str(x), self.vectors[i]))))