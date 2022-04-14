import numpy as np

from functions.process_pickle import load_pickle, save_pickle
from functions.compute_ap import get_TP_FP_FN_index


def get_sub_detections(det, label, index):
    sub_det = []
    for i in range(len(det)):
        idx = np.where(det[i][0]['class_ids'] == label)[0]
        filename = det[i][0]['filename']
        rois = det[i][0]['rois'][idx]
        scs = det[i][0]['scores'][..., idx]
        class_ids = det[i][0]['class_ids'][idx]
        masks = det[i][0]['masks'][:, :, idx]
        try:
            contents = {'filename': filename, 'rois': rois[index[i]],
                        'scores': scs[..., index[i]], 'class_ids': class_ids[index[i]],
                        'masks': masks[:, :, index[i]]}
            sub_det.append([contents])
        except Exception as e:
            print(e)
            print(i)
            print(idx)
            print(filename)
            print(rois)
            print(index[i])
            print(len(det[i][0]['rois']))
            print(len(det[i][0]['rois'][idx]))
            print()
    return sub_det


def get_sub_gts(gts, label, index):
    sub_gts = []
    for i in range(len(gts)):
        gt_index = np.where(gts[i][0]['gt_class_id'] == label)[0]
        filename = gts[i][0]['filename']
        gt_boxes = gts[i][0]['gt_bbox'][gt_index]
        gt_class_id = gts[i][0]['gt_class_id'][gt_index]
        gt_mask = gts[i][0]['gt_mask'][:, :, gt_index]
        contents = {'filename': filename, 'gt_bbox': gt_boxes[index[i]],
                    'gt_class_id': gt_class_id[index[i]], 'gt_mask': gt_mask[index[i]]}
        sub_gts.append([contents])
    return sub_gts


class_id = 2
detections_path = '../results/logs_dataset_tooth_denticle_without_handpicked_20201030/0080/detections_training_data.pkl'
gts_path = '../results/gts/gts_training_data.pkl'

detections = load_pickle(detections_path)
gts = load_pickle(gts_path)

TP_idx, FP_idx, FN_idx = get_TP_FP_FN_index(detections, gts, class_id)
print('TP: ', TP_idx[1])
print('FP: ', FP_idx[1])
print('FN: ', FN_idx[1])
print()

print('TP')
TP = get_sub_detections(detections, class_id, TP_idx)
print('FP')
FP = get_sub_detections(detections, class_id, FP_idx)
print('FN')
FN = get_sub_gts(gts, class_id, FN_idx)

print(len(TP))
print(len(FP))
print(len(FN))

save_pickle(TP, '../results/TP_denticle.pkl')
save_pickle(FP, '../results/FP_denticle.pkl')
save_pickle(FN, '../results/FN_denticle.pkl')
