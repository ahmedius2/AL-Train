from .detector3d_template import Detector3DTemplate
import torch
import time
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Final
from ..model_utils.valor_utils import *

class CenterPointMURAL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        rd = model_cfg.get('RESOLUTION_DIV', [1.0])
        self.model_cfg.VFE.RESOLUTION_DIV = rd
        self.bb3d_exist = ('BACKBONE_3D' in self.model_cfg)
        if self.bb3d_exist:
            self.model_cfg.BACKBONE_3D.RESOLUTION_DIV = rd
        self.model_cfg.MAP_TO_BEV.RESOLUTION_DIV = rd
        self.model_cfg.BACKBONE_2D.RESOLUTION_DIV = rd
        self.model_cfg.DENSE_HEAD.RESOLUTION_DIV = rd

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 0

        allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        torch.cuda.manual_seed(0)

        self.module_list = self.build_networks()

        if self.bb3d_exist:
            self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d, self.dense_head = self.module_list
        else:
            print(len(self.module_list))
            self.vfe, self.map_to_bev, self.backbone_2d, self.dense_head = self.module_list

        self.resolution_dividers = rd
        self.num_res = len(self.resolution_dividers)
        self.latest_losses = [0.] * self.num_res
        self.res_aware_1d_batch_norms, self.res_aware_2d_batch_norms = get_all_resawarebn(self)
        self.res_idx = 0

        self.inf_res_idx = self.model_cfg.get('INF_RES_INDEX', 0)
        self.res_queue = list(range(self.num_res))

    def forward_once(self, batch_dict):
        resdiv = self.resolution_dividers[self.res_idx]
        batch_dict['resolution_divider'] = resdiv
        self.vfe.adjust_voxel_size_wrt_resolution(self.res_idx)
        self.map_to_bev.adjust_grid_size_wrt_resolution(self.res_idx)
        set_bn_resolution(self.res_aware_1d_batch_norms, self.res_idx)
        set_bn_resolution(self.res_aware_2d_batch_norms, self.res_idx)
        self.dense_head.adjust_voxel_size_wrt_resolution(resdiv)

        batch_dict = self.vfe(batch_dict)
        if self.bb3d_exist:
            batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        if self.training:
            loss, tb_dict = self.get_training_loss()
            return loss, tb_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def forward(self, batch_dict):
        """
        Main forward method that handles both training and inference modes.
        For training, it accumulates losses from all resolutions and does a single backpropagation.
        """
        if self.training:
            # Initialize containers for losses and statistics
            tb_dict_combined = {}
            disp_dict = {}
            
            losses = [0.] * self.num_res
            keys = list(batch_dict.keys())
            gt_boxes_copy = batch_dict['gt_boxes']

            for ridx in self.res_queue:
                self.res_idx = ridx

                new_bd = {k:batch_dict[k] for k in keys}
                new_bd['gt_boxes'] = gt_boxes_copy.clone()

                # Forward pass for this resolution
                loss, curr_tb_dict = self.forward_once(new_bd)
                scaled_loss = loss / self.num_res
                scaled_loss.backward()

                losses[ridx] = scaled_loss.detach()

                # Store metrics with resolution prefix for logging
                res_prefix = f'res_{ridx}_'
                for k, v in curr_tb_dict.items():
                    tb_dict_combined[res_prefix + k] = v

            total_loss = sum(losses)
            ret_dict = {'loss': total_loss} # loss wont be used for backward
            return ret_dict, tb_dict_combined, disp_dict
        else:
            # For inference, just use the first resolution
            self.res_idx = self.inf_res_idx
            pred_dicts, recall_dicts = self.forward_once(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
            """
            Calculate training losses while preserving gradient information
            """
            # Get the loss tensor from the dense head
            losses, tb_dict = self.dense_head.get_loss()
            
            return losses, tb_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
