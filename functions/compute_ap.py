import numpy as np


# TODO: modify (TypeError)
def compute_overlap(boxes, query_boxes):
    """
    :param boxes: (H, W, N) ndarray of float
    :param query_boxes: (H, W, K) ndarray of float
    :return: overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    intersection = np.zeros((boxes.shape[2]))
    union = np.zeros((boxes.shape[2], query_boxes.shape[2]))
    for index in range(boxes.shape[2]):
        mask = boxes[..., index]
        intersection[index] = np.sum(np.count_nonzero(query_boxes[..., 0] & mask, axis=0), axis=0)
        union[index] = np.sum(np.count_nonzero(query_boxes[..., 0] + mask, axis=0), axis=0)
    overlaps = intersection / union
    return overlaps


def bb_intersection_over_union(gt_boxes, pred_box):
    """
    :param gt_boxes: (N, (x1, y1, x2, y2))
    :param pred_box: (x1, y1, x2, y2)
    :return:
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    ious = []
    for i in range(len(gt_boxes)):
        x_min = max(gt_boxes[i, 0], pred_box[0])
        y_min = max(gt_boxes[i, 1], pred_box[1])
        x_max = min(gt_boxes[i, 2], pred_box[2])
        y_max = min(gt_boxes[i, 3], pred_box[3])

        # compute the area of intersection rectangle
        intersection_area = max(0, x_max-x_min+1) * max(0, y_max-y_min+1)

        # compute the area of both the prediction and ground-truth rectangles
        gt_area = (gt_boxes[i, 2]-gt_boxes[i, 0]+1) * (gt_boxes[i, 3]-gt_boxes[i, 1]+1)
        pred_area = (pred_box[2]-pred_box[0]+1) * (pred_box[3]-pred_box[1]+1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # area - the intersection area
        iou = intersection_area / float(gt_area + pred_area - intersection_area)
        ious.append(iou)
    # return the intersection over union value
    return np.array(ious)


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    :param recall: The recall curve (list)
    :param precision: The precision curve (list)
    :return: The average precision as computed in py-faster-rcnn
    """
    # correct AP calculation
    # first append sentinel values at the end
    m_recall = np.concatenate(([0.], recall, [1.]))
    m_precision = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(m_precision.size - 1, 0, -1):
        m_precision[i-1] = np.maximum(m_precision[i-1], m_precision[i])

    # to calculate area under PR curve, look for points
    # when X axis (recall) changes value
    index = np.where(m_recall[1:] != m_recall[:-1])[0]

    # and sum (/Delta recall) * precision
    ap = np.sum((m_recall[index+1] - m_recall[index]) * m_precision[index+1])
    return ap


def _compute_f_score(recall, precision):
    f_score = np.zeros([len(recall)])
    index = np.where(recall + precision != 0)[0]
    f_score[index] = 2 * recall[index] * precision[index] / (recall[index] + precision[index])
    return f_score


# TODO: clean up
def get_TP_FP_FN_index(det, gt, label, iou_threshold=0.5, mode='box'):
    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    scores = np.zeros((0,))
    TP_index = []
    FP_index = []
    FN_index = []
    for i in range(len(det)):
        index = np.where(det[i][0]['class_ids'] == label)[0]
        detections = det[i][0]['rois'][index]
        masks = det[i][0]['masks'][..., index]
        scs = det[i][0]['scores'][..., index]

        gt_index = np.where(gt[i][0]['gt_class_id'] == label)[0]
        annotations = gt[i][0]['gt_class_id'][gt_index]
        gt_masks = gt[i][0]['gt_mask'][..., gt_index]
        gt_boxes = gt[i][0]['gt_bbox'][gt_index]
        detected_annotations = []
        false_negatives = np.array([gidx for gidx in range(len(gt_index))])

        for idx, (d, sc) in enumerate(zip(detections, scs)):
            scores = np.append(scores, sc)

            if annotations.shape[0] == 0:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)
                continue

            if mode == 'box':
                box = d[:4].astype(int)
                overlaps = bb_intersection_over_union(gt_boxes, box)
            else:
                mask = masks[..., idx]
                mask = np.reshape(mask > 0.5, (-1, mask.shape[-1], 1)).astype(np.uint8)  # (H, W, 1)
                overlaps = compute_overlap(gt_masks, mask)

            if overlaps.shape == ():
                overlaps = np.array([0])

            if mode == 'box':
                assigned_annotation = np.argmax(overlaps)
                max_overlap = overlaps[assigned_annotation]
            else:
                assigned_annotation = np.argmax(overlaps[0])
                max_overlap = overlaps[0][assigned_annotation]

            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                true_positives = np.append(true_positives, idx)
                detected_annotations.append(assigned_annotation)

            else:
                false_positives = np.append(false_positives, idx)

        false_negatives = np.setdiff1d(false_negatives, np.array(detected_annotations))
        TP_index.append(true_positives.astype(dtype=np.int))
        FP_index.append(false_positives.astype(dtype=np.int))
        FN_index.append(false_negatives.astype(dtype=np.int))
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
    TP_index = np.array(TP_index)
    FP_index = np.array(FP_index)
    FN_index = np.array(FN_index)
    return TP_index, FP_index, FN_index

# TODO: clean up
def evaluate(det, gt, label, iou_threshold=0.5, mode='box'):
    """
    :param det: detections
    :param gt: grand-truth
    :param label: class
    :param iou_threshold: default: 0.5
    :param mode: 'box' or 'mask' default: 'box'
    :return:
    """
    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    scores = np.zeros((0,))
    num_annotations = 0.0
    for i in range(len(det)):
        index = np.where(det[i][0]['class_ids'] == label)[0]
        detections = det[i][0]['rois'][index]
        masks = det[i][0]['masks'][..., index]
        scs = det[i][0]['scores'][..., index]

        gt_index = np.where(gt[i][0]['gt_class_id'] == label)[0]
        annotations = gt[i][0]['gt_class_id'][gt_index]
        gt_masks = gt[i][0]['gt_mask'][..., gt_index]
        gt_boxes = gt[i][0]['gt_bbox'][gt_index]
        num_annotations += annotations.shape[0]
        detected_annotations = []

        for idx, (d, sc) in enumerate(zip(detections, scs)):
            scores = np.append(scores, sc)

            if annotations.shape[0] == 0:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)
                continue

            if mode == 'box':
                box = d[:4].astype(int)
                overlaps = bb_intersection_over_union(gt_boxes, box)
            else:
                mask = masks[..., idx]
                mask = np.reshape(mask > 0.5, (-1, mask.shape[-1], 1)).astype(np.uint8)  # (H, W, 1)
                overlaps = compute_overlap(gt_masks, mask)

            if overlaps.shape == ():
                overlaps = np.array([0])

            if mode == 'box':
                assigned_annotation = np.argmax(overlaps)
                max_overlap = overlaps[assigned_annotation]
            else:
                assigned_annotation = np.argmax(overlaps[0])
                max_overlap = overlaps[0][assigned_annotation]

            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
            else:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)

    # sort by score
    indices = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    # compute average precision
    ap = _compute_ap(recall, precision)
    f_score = _compute_f_score(recall, precision)
    return ap, precision, recall, f_score, true_positives, false_positives
