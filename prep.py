import os, argparse, logging, jpype, random, gzip, shutil, zlib
from collections import defaultdict
from itertools import groupby
import numpy as np
from jpype import *
from utils import Vocab, WordVector, load_from_html, load_from_query_file, load_train_test_file, \
    save_train_test_file, load_judge_file, load_run_file, load_query_log, save_judge_file, \
    save_query_file
from config import CONFIG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('-a', '--action', help='action', type=str, default=None)
    parser.add_argument('-m', '--max_vocab_size', help='max vocabulary size', type=int, default=50000)
    parser.add_argument('-d', '--data_dir', help='data directory', type=str)
    parser.add_argument('-l', '--query_log', help='the filepath of query log', type=str)
    parser.add_argument('-r', '--train_test_ratio', help='the ratio of train and test dataset',
                        type=float, default=0.8)
    parser.add_argument('-w', '--word_vector_path', help='the filepath of word vector', type=str)
    parser.add_argument('-D', '--debug', help='whether to use debug log level', action='store_true')
    parser.add_argument('--shuqi_bing_web_dir', help='shuqi\'s html dir', type=str)
    parser.add_argument('--min_query_freq', help='minimum query frequency', type=int, default=100)
    parser.add_argument('--judgement_refer', help='judgment referred to', type=str)
    parser.add_argument('-f', '--format', help='format of input data. \
        "ir" for original format and "text" for new text matching format', type=str, default='ir')
    parser.add_argument('--reverse', help='whether to reverse the pairs in training testing files', 
        action='store_true')
    parser.add_argument('--binary_html', help='whether to read html in binary', action='store_true')
    parser.add_argument('--gzip_files', help='filepath of gzip files', type=str)
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


def load_from_html_cascade(filename, binary=False):
    try:
        result = load_from_html(filename, binary=binary)
    except jpype.JException(java.lang.StackOverflowError) as e:
        logging.warn('boilerpipe exception: {}'.format(filename))
        result = load_from_html(filename, use_boilerpipe=False, binary=binary)
    return result


def preprocess():
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
        doc_body = load_from_html_cascade(os.path.join(docs_dir, docid + '.html'), binary=binary)['body']
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
                doc_body = load_from_html_cascade(os.path.join(docs_dir, docid + '.html'), binary=binary)['body']
            if len(doc_body) == 0:
                empty_docid.add(docid)
                continue
            fp.write('{}\t{}\n'.format(docid, ' '.join(map(lambda x: str(x), vocab.encode(doc_body)))))
    print('have {} empty query, have {} empty doc'.format(len(empty_qid), len(empty_docid)))
    filter_samples(train_filepath, '{}.prep.{}'.format(*train_filepath.rsplit('.', 1)), empty_qid, empty_docid)
    filter_samples(test_filepath, '{}.prep.{}'.format(*test_filepath.rsplit('.', 1)), empty_qid, empty_docid)


def word_vector_transform():
    print('loading word vector ...')
    wv = WordVector(filepath=args.word_vector_path, first_line=True)
    vocab = Vocab(filepath=os.path.join(args.data_dir, 'vocab'), file_format='ir')
    print('transforming ...')
    wv.transform(vocab.get_word_list(), oov_filepath=os.path.join(args.data_dir, 'oov.txt'), 
        oov_at_end=True)
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


def gen_pairwise():
    train_pointwise = os.path.join(args.data_dir, 'train.prep.pointwise')
    test_pointwise = os.path.join(args.data_dir, 'test.prep.pointwise')
    for fn in [train_pointwise, test_pointwise]:
        fn_out = fn.rsplit('.', 1)[0] + '.pairwise'    
        samples = load_train_test_file(fn, file_format=args.format, reverse=args.reverse)
        samples_gb_q = groupby(samples, lambda x: x[0])
        with open(fn_out, 'w') as fout:
            for q, q_samples in samples_gb_q:
                q_samples = list(q_samples)
                for s1 in q_samples:
                    for s2 in q_samples:
                        if s1[2] > s2[2]:
                            fout.write('{}\t{}\t{}\t{}\n'.format(s1[0], s1[1], s2[1], s1[2]-s2[2]))


if __name__ == '__main__':
    if args.action == 'prep':
        preprocess()
    elif args.action == 'gen':
        generate_train_test()
    elif args.action == 'w2v':
        word_vector_transform()
    elif args.action == 'prep_query_log':
        prep_query_log()
    elif args.action == 'filter_query':
        filter_query()
    elif args.action == 'shuqi_bing_redirect':
        shuqi_bing_redirect()
    elif args.action == 'click_to_rel':
        click_to_rel()
    elif args.action == 'filter_judgement':
        filter_judgement()
    elif args.action == 'find_gzip':
        find_gzip()
    elif args.action == 'ungzip':
        ungzip()
    elif args.action == 'handle_windows':
        handle_windows()
    elif args.action == 'gen_pairwise':
        gen_pairwise()
    else:
        raise NotImplementedError('action not supported')
