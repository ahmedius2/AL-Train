import torch
import torch.nn.functional as F
import numpy as np
import datetime
import sys
import tqdm

from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou_bev
from visual_utils.bev_visualizer import visualize_bev_detections
from scipy.optimize import linear_sum_assignment
from respred_model import ResPredictor

torch.set_printoptions(precision=2, threshold=10000, sci_mode=False)

NUM_CLASSES = 10
NUM_FEATURES = 25
VISUALIZE = False

def save_checkpoint(model, optimizer, epoch, loss, test_loss, accuracy, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'test_loss': test_loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


# best is swapping train and val splits
def get_dataset(cfg, training=True, num_workers=4, batch_size=4):
    log_file = './logs/log_eval_%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = log_file + str(np.random.randint(0, 9999)) + '.txt'
    logger = common_utils.create_logger(log_file, rank=0)
    dataset, loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=batch_size,
        dist=False, workers=num_workers, logger=logger, training=training
    )

    return logger, dataset, loader, sampler

def build_model(cfg_file, ckpt_file, batchsz=1, training=True):
    cfg_from_yaml_file(cfg_file, cfg)
    
    set_cfgs = ['OPTIMIZATION.BATCH_SIZE_PER_GPU', str(batchsz),
        'MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH', '0.3',
        'DATA_CONFIG.VERSION', 'v1.0-trainval']

    cfg_from_list(set_cfgs, cfg)
    logger, dataset, loader, sampler = get_dataset(cfg, training, batchsz, batchsz)
    print(f'Loaded dataset with {len(dataset)} samples')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=False)

    return model, loader


def forward_to_get_inp_and_outp(model, batch_dict):
    sample_tokens = [mt['token'] for mt in batch_dict['metadata']]

    # We could execute the following part in a seperate process
    #pred_exec_times_ms = [122, 96, 78]
    pred_exec_times_ms = [300, 200, 100]
    gt_boxes_per_res = [None] * model.num_res
    for res_idx in range(model.num_res):
        cur_gt_boxes = []
        for st in sample_tokens:
            tdiff_musec = pred_exec_times_ms[res_idx] * 1000
            cur_gt_boxes.append(model.dataset.get_moved_gt_boxes(
                    st, tdiff_musec))
        gt_boxes_per_res[res_idx] = cur_gt_boxes

    load_data_to_gpu(batch_dict)

    pred_dicts_per_res = [None] * model.num_res
    for res_idx in range(model.num_res):
        model.inf_res_idx = res_idx
        pred_dicts_per_res[res_idx], ret_dict = model(batch_dict)

    # Calculate the prediction score for each batch
    ne = len(sample_tokens)
    res_detscores = []
    for res_idx, pred_dicts in enumerate(pred_dicts_per_res):
        batch_gt_boxes = gt_boxes_per_res[res_idx]
        cur_detscores = []
        for batch_idx, (pred_dict, gt_boxes_) in enumerate(zip(pred_dicts, batch_gt_boxes)):
            pred_boxes = pred_dict['pred_boxes'][:, :7].clone() # ignore velocity
            gt_boxes = gt_boxes_.copy()

            if gt_boxes.shape[0] == 0:
                return None, None, None

            # we need to go this to match with ground truth
            pred_boxes_w = pred_boxes[:, 3].clone()
            pred_boxes[:, 3] = pred_boxes[:, 4]
            pred_boxes[:, 4] = pred_boxes_w

            if VISUALIZE:
                st = sample_tokens[batch_idx]
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

            cost_matrix = boxes_iou_bev(pred_boxes, gt_boxes).cpu().numpy()
            pred_indices, gt_indices = linear_sum_assignment(-cost_matrix)

            total_iou = 0.
            for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                iou = cost_matrix[pred_idx, gt_idx]
                total_iou += iou
            fp = len(pred_boxes) - len(pred_indices)
            score =  total_iou / (len(gt_boxes) + fp)

#            # Use Hungarian algorithm for matching, returns matched inds
#            # NOTE negate cost matrix when using IOU
#            cost_matrix = torch.cdist(pred_boxes[:, :2], gt_boxes[:, :2]).cpu().numpy()
#            pred_indices, gt_indices = linear_sum_assignment(-cost_matrix)
#            dist_thresholds = [0.5, 1.0, 2.0, 4.0]
#            matches = np.zeros(4)
#            for pred_idx, gt_idx in zip(pred_indices, gt_indices):
#                dist = cost_matrix[pred_idx, gt_idx]
#                for i, thr in enumerate(dist_thresholds):
#                    if dist <= thr:
#                        matches[i] += 1
#            mean_precision = np.mean(matches / len(pred_boxes))
#            mean_recall =  np.mean(matches / len(gt_boxes))
#            score = (mean_precision + mean_recall)/2
            cur_detscores.append(score)
        res_detscores.append(cur_detscores)

    highest_res_pred_dicts = pred_dicts_per_res[0]
    batch_size = len(highest_res_pred_dicts)
    max_dist = np.max(np.absolute(model.dataset.point_cloud_range))
    max_num_obj = max((pd['pred_labels'].size(0) for pd in highest_res_pred_dicts))
    input_tensor = torch.zeros((batch_size, max_num_obj, NUM_FEATURES), device='cuda')
    mask = torch.zeros((batch_size, max_num_obj), dtype=torch.bool, device='cuda')
    for batch_idx, pred_dict in enumerate(highest_res_pred_dicts):
        # Each pred_dict is an element of batch
        boxes = pred_dict['pred_boxes']
        labels = pred_dict['pred_labels']
        scores = pred_dict['pred_scores']

        num_obj = labels.size(0)
        mask[batch_idx, :num_obj] = True

        boxes[:, :3] /= max_dist

        dist_norm = torch.linalg.vector_norm(boxes[:, :2], dim=1, keepdim=True)
        egovel = torch.from_numpy(model.dataset.get_egovel(sample_tokens[batch_idx])) / max_dist
        egovel = egovel.cuda()

        # make the velocities relative velocity
        boxes[:, 7:9] = (boxes[:, 7:9] / max_dist) - egovel[:2]

        # velocity vector norms
        vel_norm = torch.linalg.vector_norm(boxes[:, 7:9], dim=1, keepdim=True)

        # Calculate the distance each object would travel regarding execution time of each res
        dists_to_travel = [vel_norm * (etime_ms/1000) \
                for etime_ms in pred_exec_times_ms]
        dists_to_travel = torch.cat(dists_to_travel, dim=1)
        label_encodings = F.one_hot(labels.long()-1, num_classes=NUM_CLASSES) #NOTE ensure its right
        input_tensor[batch_idx, :num_obj] = torch.cat((boxes, dist_norm, vel_norm, dists_to_travel,
                scores.unsqueeze(-1), label_encodings), dim=1)

    res_detscores = torch.tensor(res_detscores, dtype=torch.float).T
    mins = torch.min(res_detscores, dim=1, keepdim=True).values
    maxs = torch.max(res_detscores, dim=1, keepdim=True).values
    res_detscores = (res_detscores - mins) / (maxs - mins)
    if torch.isnan(res_detscores.flatten()).any():
        # skip this guy
        return None, None, None

    return input_tensor, mask, res_detscores.cuda()

if __name__ == '__main__':
    cfg_file="./cfgs/nuscenes_models/mural_pillarnet_016_020_032.yaml"
    ckpt_file="../output/nuscenes_models/mural_pillarnet_016_020_032/default/ckpt/checkpoint_epoch_20.pth"

    batch_size=2
    #model, dataloader = build_model(cfg_file, ckpt_file, batchsz=batch_size, training=True)
    #model.eval() # run with @torch.no_grad
    #model.dataset.disable_data_augmentor()
    #model.cuda()

    test_model, test_dataloader = build_model(cfg_file, ckpt_file, batchsz=batch_size, training=False)
    test_model.eval() # run with @torch.no_grad
    #model.dataset.disable_data_augmentor()
    test_model.cuda()

    res_predictor = ResPredictor(input_dim=NUM_FEATURES, num_detectors=test_model.num_res)
    res_predictor.train()
    res_predictor.cuda()

    num_epochs = 20
    lr = 0.001
    optimizer = torch.optim.Adam(res_predictor.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        res_predictor.train()
        train_loss = 0.0
        samples_processed = 0

        progress_bar = tqdm.tqdm(total=len(test_dataloader)//2, leave=True, desc='train', dynamic_ncols=True)
        for batch_idx, batch_dict in enumerate(test_dataloader):
            if batch_idx >= len(test_dataloader)//2:
                break
            with torch.no_grad():
                input_tensor, mask, scores = forward_to_get_inp_and_outp(test_model, batch_dict)
            if input_tensor is None:
                continue

            preds = res_predictor(input_tensor, mask)
            loss = criterion(preds, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            samples_processed +=1

            progress_bar.set_postfix({'loss': train_loss/samples_processed})
            progress_bar.update()
        progress_bar.close() 

        progress_bar = tqdm.tqdm(total=len(test_dataloader)//2, leave=True, desc='test', dynamic_ncols=True)
        total_test_loss = 0
        true_positives = 0
        samples_processed = 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(test_dataloader):
                if batch_idx <= len(test_dataloader)//2:
                    continue

                input_tensor, mask, scores = forward_to_get_inp_and_outp(test_model, batch_dict)
                if input_tensor is None:
                    continue

                preds = res_predictor(input_tensor, mask)
                test_loss = criterion(preds, scores)
                total_test_loss += test_loss.item()

                pred_max_indices = torch.argmax(preds, dim=1)
                label_max_indices = torch.argmax(scores, dim=1)
                true_positives += (pred_max_indices == label_max_indices).sum().item()
                samples_processed +=1
                progress_bar.set_postfix({'test_loss': total_test_loss/samples_processed})
                progress_bar.update()

            accuracy = true_positives / (samples_processed * batch_size)
            print(f'Accuracy: %{accuracy * 100}')
        progress_bar.close() 

        #del test_model
        #del test_dataloader

        filename=f"../output/nuscenes_models/res_predictor/checkpoint_epoch{epoch+1}.pth"
        save_checkpoint(res_predictor, optimizer, epoch, loss, test_loss, accuracy, filename)
