import numpy as np


# TODO: modify (TypeError)
def compute_overlap(masks, query_masks):
    """
    :param masks: (H, W, N) ndarray of float
    :param query_masks: (H, W, K) ndarray of float
    :return overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    intersection = np.zeros((masks.shape[2]))
    union = np.zeros((masks.shape[2], query_masks.shape[2]))
    for index in range(masks.shape[2]):
        mask = masks[..., index]
        intersection[index] = np.sum(np.count_nonzero(query_masks[..., 0] & mask, axis=0), axis=0)
        union[index] = np.sum(np.count_nonzero(query_masks[..., 0] + mask, axis=0), axis=0)
    overlaps = intersection / union
    return overlaps
