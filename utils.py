import contextlib, sys, re, logging, time, html, jieba, json
from collections import defaultdict
from collections import namedtuple
import numpy as np
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from bs4 import BeautifulSoup, UnicodeDammit
from xml.etree import ElementTree
import xml.etree.ElementTree as ET
from lxml import etree
from nltk.tokenize import word_tokenize


PointwiseSample = namedtuple('PointwiseSample', 'qid docid label query doc')
PairwiseSample = namedtuple('PairwiseSample', 'qid docid1 docid2 label query doc1 doc2')


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def qc_xml_field_line_map(filepath, field=None, map_fn=lambda x:x, ind=None, start=None, end=None):
    '''
    Iterate through the file line by line and map the selected lines.
    '''
    if ind is not None:
        out_filepath = '{}.{}{}'.format(filepath, 'seg', ind)
    else:
        out_filepath = '{}.{}'.format(filepath, 'seg')
    with open(filepath, 'r') as fin, open(out_filepath, 'w') as fout:
        for i, l in enumerate(fin):
            if ind is not None and i < start:
                continue
            if ind is not None and i >= end:
                break
            if i % 500000 == 0:
                print('{}w'.format(i//10000))
            if type(field) is tuple and l.lstrip().startswith(field):
                start_ind = l.find('>') + 1
                end_ind = l.rfind('</')
                prefix = l[:start_ind]
                suffix = l[end_ind:]
                #ele = ET.fromstring(l)
                #ele.text = map_fn(ele.text)
                #l = prefix + ElementTree.tostring(ele, encoding='utf-8', method='xml').decode('utf-8') + '\n'
                l = prefix + map_fn(l[start_ind:end_ind]) + suffix
            fout.write(l)


def parse_sogou_qcl_one_qd(lines):
    struct = ('q', ['query', 'query_frequency', 'query_id',
                    ('doc', ['url', 'doc_id', 'title', 'content', 'html', 'doc_frequency',
                             ('relevance', ['TCM', 'DBN', 'PSCM', 'TACM', 'UBM'])])])
    def recursive_parse(lines, start, struct, result):
        i = start
        while True:
            if i >= len(lines):
                break
            if lines[i].strip() != '<{}>'.format(struct[0]):
                break
            one_result = {}
            i += 1
            for j, tag in enumerate(struct[1]):
                if type(tag) is str:
                    line = lines[i].strip()
                    # parse
                    if not line.startswith('<{}>'.format(tag)) or not line.endswith('</{}>'.format(tag)):
                        print(lines[i])
                        raise Exception('parse error')
                    one_result[tag] = line[line.find('>')+1:line.rfind('<')]
                    i += 1
                elif type(tag) is tuple:
                    # go deep
                    recursive_result = []
                    one_result[tag[0]] = recursive_result
                    i = recursive_parse(lines, i, tag, recursive_result)
            if lines[i].strip() != '</{}>'.format(struct[0]):
                print(lines[i])
                raise Exception('parse error')
            result.append(one_result)
            i += 1
        return i
    result = []
    recursive_parse(lines, 0, struct, result)
    return result


def qd_xml_filter(filepath, out_filepath, filter_fn=lambda x: False):
    '''
    Iterate over <q></q> objects and filter.
    '''
    total = 0
    remain = 0
    with open(filepath, 'r') as fin, open(out_filepath, 'w') as fout:
        lines = []
        for l in fin:
            if l.startswith('<q>'):
                lines = [l]
            elif l.startswith('</q>'):
                lines.append(l)
                qd = parse_sogou_qcl_one_qd(lines)
                total += 1
                if not filter_fn(qd[0]):
                    remain += 1
                    for line in lines:
                        fout.write(line)
            else:
                lines.append(l)
    print('{} out of {} are preserved'.format(remain, total))


def qd_xml_iterator(filepath):
    '''
    Iterate over <q></q> objects and return one at a time.
    '''
    #parser = ET.XMLParser(encoding="utf-8")
    #parser = etree.XMLParser(recover=True)
    with open(filepath, 'r') as fin:
        lines = []
        for l in fin:
            if l.startswith('<q>'):
                lines = [l]
            elif l.startswith('</q>'):
                lines.append(l)
                #yield ET.fromstring(''.join(lines), parser=parser)
                #yield etree.fromstring(''.join(lines), parser=parser)
                qd = parse_sogou_qcl_one_qd(lines)
                #with open('test.json', 'w') as outfile:
                #    json.dump(qd, outfile, indent=True, ensure_ascii=False)
                yield qd[0]
            else:
                lines.append(l)


def qd_xml_to_prep(from_file, prep_file, vocab, field_in_xml='title', relevance='TACM'):
    print('generate {}'.format(prep_file))
    count_q, count_d = 0, 0
    with open(prep_file, 'w') as prep_f:
        for i, qd in enumerate(qd_xml_iterator(from_file)):
            '''
            qid = qd.find('./query_id').text
            query = qd.find('./query').text
            '''
            qid = qd['query_id']
            query = qd['query']
            if len(query.strip()) == 0:
                continue
            count_q += 1
            query = vocab.encode(query.split(' '))
            query = ' '.join(map(str, query))
            # for doc in qd.findall('./doc'):
            for doc in qd['doc']:
                '''
                docid = doc.find('./doc_id').text
                doc_text = doc.find('./{}'.format(field_in_xml)).text
                label = doc.find('./relevance/{}'.format(relevance)).text
                '''
                docid = doc['doc_id']
                doc_text = doc[field_in_xml]
                label = doc['relevance'][0][relevance]
                if len(doc_text.strip()) == 0:
                    continue
                count_d += 1
                doc_text = vocab.encode(doc_text.split(' '))
                doc_text = ' '.join(map(str, doc_text))
                prep_f.write('{}\t{}\t{}\t{}\t{}\n'.format(qid, docid, label, query, doc_text))
        print('totally {} queries, {} docs'.format(count_q, count_d))


def word_segment(text):
    return list(jieba.cut(text, cut_all=False))


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


def load_judge_file(filepath, scale=int, file_format='ir', reverse=False):
    qd_judge = defaultdict(lambda: defaultdict(lambda: None))
    with open(filepath, 'r') as fp:
        for l in fp:
            if file_format == 'ir':
                q, d, r = l.rstrip().split('\t')[:3]
            elif file_format == 'text':
                r, q, d = l.rstrip().split(' ')[:3]
            r = scale(r)
            if reverse:
                qd_judge[d][q] = r
            else:
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


def save_prep_file(filepath, docs, file_format='ir'):
    with open(filepath, 'w') as fout:
        for ind, d in docs:
            if file_format == 'ir':
                fout.write('{}\t{}\n'.format(ind, ' '.join(map(lambda x: str(x), d))))
            elif file_format == 'text':
                fout.write('{} {}\n'.format(ind, ' '.join(map(lambda x: str(x), d))))
            else:
                raise Exception()


def load_prep_file_aslist(filepath, file_format='ir', use_split=False, func=int):
    '''
    load the word sequence with or without splitting.
    '''
    result = []
    with open(filepath, 'r') as fp:
        for i, l in enumerate(fp):
            l = l.rstrip('\n')
            if len(l) == 0:
                continue
            if file_format == 'ir':
                k, ws = l.split('\t', 1)
            elif file_format == 'text':
                k, ws = l.split(' ', 1)
            else:
                raise Exception()
            if use_split:
                ws = [func(w) for w in ws.split(' ') if len(w) > 0]
            result.append((k, ws))
    return result


def load_prep_file(filepath, file_format='ir', func=int):
    result = {}
    with open(filepath, 'r') as fp:
        for l in fp:
            l = l.rstrip('\n')
            if len(l) == 0:
                continue
            if file_format == 'ir':
                k, ws = l.split('\t')
                result[k] = [func(w) for w in ws.split(' ') if len(w) > 0]
            elif file_format == 'text':
                ws = l.split(' ')
                k = ws[0]
                ws = ws[1:]
                result[k] = [func(w) for w in ws]
            else:
                raise Exception()
    return result


def prep_file_mapper(filepath, out_filepath, method='sample', func=int, map_fn=lambda x: x):
    if filepath.endswith('pointwise'):
        mode = 1
    elif filepath.endswith('pairwise'):
        mode = 2
    else:
        raise ValueError('not supported')
    with open(out_filepath, 'w') as fout:
        for samples in prep_file_iterator(filepath, method=method, func=func, parse=False):
            samples = map_fn(samples)
            if type(samples) != list:
                samples = [samples]
            for sample in samples:
                if mode == 1:
                    fout.write('{}\t{}\t{}\t{}\t{}\n'.format(
                        sample.qid, sample.docid, sample.label,
                        sample.query, sample.doc))
                elif mode == 2:
                    fout.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        sample.qid, sample.docid1, sample.docid2, sample.label,
                        sample.query, sample.doc1, sample.doc2))


def prep_file_iterator(filepath, method='sample', func=int, parse=False):
    '''
    When method='query', return all the samples of a specific query each time.
    When method='smaple', return one sample each time.
    '''
    result = []
    if filepath.endswith('pointwise'):
        mode = 1
    elif filepath.endswith('pairwise'):
        mode = 2
    else:
        raise ValueError('not supported')
    if method not in {'query', 'sample'}:
        raise ValueError('not supported')
    with open(filepath, 'r') as fin:
        for l in fin:
            l = l.strip()
            if mode == 1:
                qid, docid, label, query, doc = l.split('\t')
                label = func(label)
                if parse:
                    query = list(map(int, query.split()))
                    doc = list(map(int, doc.split()))
                sample = PointwiseSample(qid, docid, label, query, doc)
            elif mode == 2:
                qid, docid1, docid2, label, query, doc1, doc2 = l.split('\t')
                label = func(label)
                if parse:
                    query = list(map(int, query.split()))
                    doc1 = list(map(int, doc1.split()))
                    doc2 = list(map(int, doc2.split()))
                sample = PairwiseSample(qid, docid1, docid2, label, query, doc1, doc2)
            if method == 'sample':
                yield sample
            elif method == 'query':
                if len(result) > 0 and result[-1].qid != sample.qid:
                    yield result
                    result = []
                result.append(sample)
        if len(result) > 0:
            yield result


def load_train_test_file(filepath, file_format='ir', reverse=False):
    samples = []
    with open(filepath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if filepath.endswith('pointwise'):
                if file_format == 'ir':
                    q, d, r = l.split('\t')
                elif file_format == 'text':
                    r, q, d = l.split(' ')
                else:
                    raise Exception()
                if reverse:
                    samples.append((d, q, int(r)))
                else:
                    samples.append((q, d, int(r)))
            elif filepath.endswith('pairwise'):
                if file_format == 'ir':
                    q, d1, d2, r = l.split('\t')
                elif file_format == 'text':
                    q, d1, d2, r = l.split('\t')
                else:
                    raise Exception()
                samples.append((q, d1, d2, float(r)))
    return samples


def save_train_test_file(samples, filepath, file_format='ir'):
    with open(filepath, 'w') as fp:
        for s in samples:
            if file_format == 'ir':
                if filepath.endswith('pointwise'):
                    fp.write('{}\n'.format('\t'.join(str(i) for i in s)))
                elif filepath.endswith('pairwise'):
                    fp.write('{}\n'.format('\t'.join(str(i) for i in s)))
                else:
                    raise Exception()
            elif file_format == 'text':
                if filepath.endswith('pointwise'):
                    fp.write('{} {} {}\n'.format(s[2], s[0], s[1]))
                else:
                    raise Exception()


def load_word_vector(filepath, is_binary=False, first_line=True):
    if is_binary:
        raise NotImplementedError()
    words = []
    vectors = []
    with open(filepath, 'r') as fp:
        if first_line:
            vocab_size, dim = fp.readline().split()
            vocab_size = int(vocab_size)
            dim = int(dim)
            for i in range(vocab_size):
                if i % 100000 == 0:
                    print(i/10000)
                rl = fp.readline().rstrip()
                l = rl.split(' ')
                words.append(l[0])
                v = [float(f) for f in l[1:]]
                if len(v) != dim:
                    raise Exception('word vector format error')
                vectors.append(v)
        else:
            for i, l in enumerate(fp):
                if i % 100000 == 0:
                    print(i/10000)
                l = l.rstrip().split(' ')
                words.append(l[0])
                v = [float(f) for f in l[1:]]
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


def load_boilerpipe():
    from boilerpipe.extract import Extractor


def load_from_html(filename, use_boilerpipe=True, use_nltk=True, use_regex=True,
    binary=False, field=['title', 'body']):
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
    if 'title' in field:
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
    if 'body' in field:
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
    result = {}
    if 'title' in field:
        result['title'] = my_word_tokenize(title) if use_nltk else clean_text(title).split(' ')
    if 'body' in field:
        result['body'] = my_word_tokenize(body) if use_nltk else clean_text(body).split(' ')
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
    def __init__(self, max_size=None, filepath=None, file_format='ir'):
        self.UNK = '<UNK>'
        self.word2count = {}
        self.word2ind = {}
        self.vocab_size = 0
        self.max_size = max_size or sys.maxsize
        self.file_format = file_format
        if filepath != None:
            self.load_from_file(filepath)


    def get_word_list(self):
        return [self.ind2word[i] for i in range(self.vocab_size)]


    def add(self, word):
        if len(word) == 0:
            return
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
                if self.file_format == 'ir':
                    w, c = l.split('\t')
                elif self.file_format == 'text':
                    w, r, c = l.split(' ')
                else:
                    raise Exception()
                if not ind:
                    self.UNK = w
                else:
                    self.word2count[w] = int(c)
                self.word2ind[w] = ind
                ind += 1
            self.vocab_size = ind
        self.ind2word = dict(zip(self.word2ind.values(), self.word2ind.keys()))


class WordVector(object):
    def __init__(self, filepath=None, is_binary=False, first_line=True, initializer='uniform'):
        if initializer not in {'uniform'}:
            raise Exception('initializer not supported')
        self.initializer = initializer
        if filepath != None:
            self.raw_words, self.raw_vectors = load_word_vector(filepath, is_binary=is_binary, first_line=first_line)
        self.raw_vocab_size = len(self.raw_words)
        self.raw_words2ind = dict(zip(self.raw_words, range(self.raw_vocab_size)))
        self.dim = self.raw_vectors.shape[1]
        self.vocab_size = self.raw_vectors.shape[0]
        self.words = np.array(self.raw_words)
        self.vectors = np.array(self.raw_vectors)


    def transform(self, new_words, oov_filepath=None, oov_at_end=False):
        '''
        oov_at_end determines whether the oov words are appended at the end
        '''
        new_words = np.array(new_words)
        start_ind = self.raw_vocab_size
        def new_inder(w):
            nonlocal start_ind
            if w in self.raw_words2ind:
                return self.raw_words2ind[w]
            else:
                start_ind += 1
                return start_ind - 1
        new_ind = np.array([new_inder(w) for w in new_words])
        if oov_at_end:
            ind_sorted = np.argsort(new_ind)
            new_ind = new_ind[ind_sorted]
            new_words = new_words[ind_sorted]
        self.words = new_words
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


    def svd(self, n_components=10):
        svd = TruncatedSVD(n_components=n_components, algorithm='arpack')
        new_vectors = svd.fit_transform(self.vectors)
        self.dim = new_vectors.shape[1]
        self.vectors = new_vectors


    def save_to_file(self, filepath, is_binary=False):
        if is_binary:
            raise NotImplementedError()
        with open(filepath, 'w') as fp:
            fp.write('{} {}\n'.format(self.vocab_size, self.dim))
            for i in range(self.vocab_size):
                fp.write('{} {}\n'.format(self.words[i], ' '.join(map(lambda x: str(x), self.vectors[i]))))