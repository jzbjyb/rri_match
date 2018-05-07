import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import powerlaw
import logging, math
import matplotlib.pyplot as plt
from __main__ import args


POWER_LOW_A = 0.5
POWER_LOW_MAX = 100 # maximum number of matching doc terms in a matrix
PROXIMITY_ALPHA = 0.8
UNIFORM_NOISE = 1e-10


def eval_mm(mms, features='tf'):
    '''
    Evaluate the relevance of the given match matrix.
    @param mms: a numpy array of size (bs, h, w)
    @return: a numpy array of size (bs)
    '''
    features = set(features.split('_'))
    bs, h, w = mms.shape
    score = np.zeros((bs), dtype=np.float32)
    if 'tf' in features:
        tf = np.sum(np.sum(mms, axis=2), axis=1) / (h * w)
        score += tf
    if 'proximity' in features:
        doc_term_matched = np.max(mms, axis=1) >= 1
        proximity = []
        for i in range(bs):
            inds = np.array(range(w))[doc_term_matched[i]]
            dist = inds[1:] - inds[:-1]
            if len(dist) == 0:
                proximity.append(w)
            else:
                proximity.append(np.min(dist))
        proximity = np.array([np.log(np.exp(-p) + PROXIMITY_ALPHA) for p in proximity])
        score += proximity
    return score


def get_eval_mm(mms):
    bs, h, w = mms.shape
    emms = np.zeros_like(mms, dtype=np.float32)
    emms[:, h-1, w-1] = eval_mm(mms, features=args.feature)
    wmms = np.zeros_like(mms, dtype=np.float32)
    wmms[:, h-1, w-1] = 1
    return np.random.uniform(-UNIFORM_NOISE, UNIFORM_NOISE, mms.shape) + mms / w, emms, wmms


def get_one2one_mm(bs, h, w, mean_match_query_term=None):
    '''
    Get match matrix with one query term only matching one doc term.
    @param bs: batch size
    @param h: height of the matrix (number of query terms)
    @param w: width of the matrix (number of document terms)
    @return: a numpy array of size (bs, h, w)
    '''
    actual_w = max(h, w)
    mms = []
    for i in range(bs):
        data = [1] * h
        row = range(h)
        col = np.random.choice(actual_w, h, replace=False)
        mms.append(csr_matrix((data, (row, col)), shape=(h, actual_w)).toarray()[:, :w])
    match_prob = min(1, mean_match_query_term / h) if mean_match_query_term != None else 0.5
    return  get_eval_mm(np.random.choice(2, size=(bs, h, 1), p=[1-match_prob, match_prob]) * np.stack(mms))


def get_one2many_mm(bs, h, w, mean_match_query_term=None, mean_match_doc_term=None, dist='binomial'):
    '''
    Get match matrix with one query term matching many doc terms.
    @param bs: batch size
    @param h: height of the matrix (number of query terms)
    @param w: width of the matrix (number of document terms)
    @return: a numpy array of size (bs, h, w)
    '''
    if dist not in {'binomial', 'power_law', 'custom'}:
        raise Exception('not supported distribution.')
    mms = []
    for i in range(bs):
        data = [1] * w
        col = range(w)
        row = np.random.choice(h, w)
        mms.append(csr_matrix((data, (row, col)), shape=(h, w)).toarray())
    q_match_prob = min(1, mean_match_query_term / h) if mean_match_query_term != None else 0.5
    d_match_prob = min(1, mean_match_doc_term / w) if mean_match_doc_term != None else 0.5
    if dist=='custom':
        # select rows (query terms) and columns (doc terms) independently
        mask = np.random.choice(2, size=(bs, h, 1), p=[1-q_match_prob, q_match_prob]) * \
               np.random.choice(2, size=(bs, 1, w), p=[1-d_match_prob, d_match_prob])
    elif dist=='binomial':
        # the number of columns (doc terms) with match obeys binomial distribution
        # the probability of a column to be reserved (have match) is q_match_prob*d_match_prob
        mask = np.random.choice(2, size=(bs, h, w),
                                p=[1-q_match_prob*d_match_prob, q_match_prob*d_match_prob])
    elif dist=='power_law':
        # the number of columns (doc terms) with match obeys power law distribution
        d_match_nums = []
        while len(d_match_nums) < bs:
            n = math.floor(powerlaw.rvs(POWER_LOW_A) * (POWER_LOW_MAX + 1))
            if n <= w:
                d_match_nums.append(n)
        #print(np.histogram(d_match_nums, bins=range(w + 1)))
        #plt.hist(d_match_nums, bins=range(w + 1))
        #plt.show()
        def ind_to_arr(inds, len):
            arr = np.zeros((len), dtype=np.float32)
            arr[inds] = 1
            return arr
        mask = np.stack([ind_to_arr(np.random.choice(w, size=(n), replace=False), w)
                         for n in d_match_nums])
        mask = np.expand_dims(mask, axis=1)
    return get_eval_mm(mask * np.stack(mms))