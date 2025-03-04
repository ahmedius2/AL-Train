import torch
import numpy as np
import datetime
import sys

from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou_bev
from visual_utils.bev_visualizer import visualize_bev_detections
from scipy.optimize import linear_sum_assignment

VISUALIZE = False

# best is swapping train and val splits
def get_dataset(cfg, training=True, num_workers=4, batch_size=4):
    log_file = './logs/log_eval_%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = log_file + str(np.random.randint(0, 9999)) + '.txt'
    logger = common_utils.create_logger(log_file, rank=0)
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=batch_size,
        dist=False, workers=num_workers, logger=logger, training=training
    )

    return logger, test_set, test_loader, sampler

def build_model(cfg_file, ckpt_file, batchsz=1):
    cfg_from_yaml_file(cfg_file, cfg)
    
    set_cfgs = ['OPTIMIZATION.BATCH_SIZE_PER_GPU', '4',
        'MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH', '0.3',
        'DATA_CONFIG.VERSION', 'v1.0-mini']

    cfg_from_list(set_cfgs, cfg)
    logger, test_set, test_loader, sampler = get_dataset(cfg, True, batchsz, batchsz)
    print(f'Loaded dataset with {len(test_set)} samples')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=False)
    model.eval() # run with @torch.no_grad
    model.cuda()

    return model, test_loader

if __name__ == '__main__':
    cfg_file="./cfgs/nuscenes_models/mural_pillarnet_016_020_032.yaml"
    ckpt_file="../output/nuscenes_models/mural_pillarnet_016_020_032/default/ckpt/checkpoint_epoch_17.pth"

    model, dataloader = build_model(cfg_file, ckpt_file, batchsz=1)
    model.dataset.disable_data_augmentor()

    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            sample_tokens = [mt['token'] for mt in batch_dict['metadata']]

            # We could execute the following part in a seperate process
            #pred_exec_times_ms = [122, 96, 78]
            pred_exec_times_ms = [400, 200, 100]
            gt_boxes_per_res = []
            for res_idx in range(model.num_res):
                for st in sample_tokens:
                    tdiff_musec = pred_exec_times_ms[res_idx] * 1000
                    gt_boxes_per_res.append(model.dataset.get_moved_gt_boxes(
                            st, tdiff_musec))

            load_data_to_gpu(batch_dict)

            pred_dicts_per_res = [None] * model.num_res
            for res_idx in range(model.num_res):
                model.inf_res_idx = res_idx
                pred_dicts_per_res[res_idx], ret_dict = model(batch_dict)

            # Calculate the prediction score for each batch
            ne = len(sample_tokens)
            for i, pred_dicts in enumerate(pred_dicts_per_res):
                batch_gt_boxes = gt_boxes_per_res[i*ne:(i+1)*ne]
                for j, (pred_dict, gt_boxes_) in enumerate(zip(pred_dicts, batch_gt_boxes)):
                    pred_boxes = pred_dict['pred_boxes'][:, :7].clone() # ignore velocity
                    gt_boxes = gt_boxes_.clone()

                    # we need to go this to match with ground truth
                    pred_boxes_w = pred_boxes[:, 3].clone()
                    pred_boxes[:, 3] = pred_boxes[:, 4]
                    pred_boxes[:, 4] = pred_boxes_w

                    if VISUALIZE:
                        st = sample_tokens[j]
                        out_path = f"images/res{i}_{st}.png"
                        visualize_bev_detections(
                            pred_boxes.cpu().numpy(),
                            gt_boxes[:, :7], # already numpy
                            save_path=out_path,
                        )
                        print('Saved', out_path)

                    pred_labels = pred_dict['pred_labels']
                    gt_labels = torch.from_numpy(gt_boxes[:, -1]).cuda()
                    gt_boxes = torch.from_numpy(gt_boxes[:, :7]).float().cuda()
                    
                    # Ensure boxes of same class are matched by pushing 100m * label id
                    pred_boxes[:, 0] += (pred_labels * 100)
                    gt_boxes[:, 0] += (gt_labels * 100)

                    #cost_matrix = boxes_iou_bev(pred_boxes, gt_boxes).cpu().numpy()
                    cost_matrix = torch.cdist(pred_boxes[:, :2], gt_boxes[:, :2]).cpu().numpy()

                    # Use Hungarian algorithm for matching, returns matched inds
                    # NOTE negate cost matrix when using IOU
                    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

                    dist_thresholds = [0.5, 1.0, 2.0, 4.0]
                    matches = np.zeros(4)
                    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                        dist = cost_matrix[pred_idx, gt_idx]
                        for i, thr in enumerate(dist_thresholds):
                            if dist <= thr:
                                matches[i] += 1

                    mean_precision = np.mean(matches / len(pred_boxes))
                    mean_recall =  np.mean(matches / len(gt_boxes))
                    score = mean_precision + mean_recall
        sys.exit()
