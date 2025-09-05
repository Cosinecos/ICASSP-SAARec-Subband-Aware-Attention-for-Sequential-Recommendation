import numpy as np
def hit_rate(ranklist, pos_item): return 1.0 if pos_item in ranklist else 0.0
def ndcg(ranklist, pos_item):
    if pos_item in ranklist:
        idx = ranklist.index(pos_item)
        return 1.0 / np.log2(idx + 2)
    return 0.0
def evaluate_ranking(scores, gt_item, topk=(10,20)):
    result = {}
    sorted_idx = np.argsort(-scores)
    for K in topk:
        topk_items = sorted_idx[:K].tolist()
        result[K] = (hit_rate(topk_items, gt_item), ndcg(topk_items, gt_item))
    return result
