import numpy as np


def dcg(query, doc_list, qd_judge, top_k=5):
    real_top_k = min(top_k, len(doc_list))
    score = 0
    for i in range(real_top_k):
        score += (2 ** qd_judge[query][doc_list[i]] - 1) / np.log2(i + 2)
    return score


def ndcg(query, doc_list, qd_judge, top_k=5):
    un = dcg(query, doc_list, qd_judge, top_k=top_k)
    ideal = dcg(query, [d[0] for d in sorted(qd_judge[query].items(), key=lambda x: -x[1])], qd_judge, top_k=top_k)
    ideal = ideal or 1
    normalized = un / ideal
    if normalized > 1:
        raise Exception('normalized nDCG exceeds 1')
    return normalized


def precision(query, doc_list, qd_judge, top_k=5):
    top_k = min(top_k, len(doc_list))
    rels = [i for i in range(top_k) if qd_judge[query][doc_list[i]] > 0]
    return len(rels) / (top_k or 1)


def average_precision(query, doc_list, qd_judge, top_k=5):
    top_k = min(top_k, len(doc_list))
    rels = [i for i in range(top_k) if qd_judge[query][doc_list[i]] > 0]
    if len(rels) == 0:
        return 0
    ap = np.mean([(i+1)/(r+1) for i, r in enumerate(rels)])
    return ap


def evaluate(ranks, qd_judge, metric=ndcg, **kwargs):
    return dict([(q, metric(q, ranks[q], qd_judge, **kwargs)) for q in ranks])