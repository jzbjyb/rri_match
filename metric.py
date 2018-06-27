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


def evaluate(ranks, qd_judge, metric=ndcg, **kwargs):
    return dict([(q, metric(q, ranks[q], qd_judge, **kwargs)) for q in ranks])