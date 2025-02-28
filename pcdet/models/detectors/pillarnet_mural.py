from .detector3d_template import Detector3DTemplate
import torch
import time
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Final
from ..model_utils.valor_utils import *

class PillarNetMURAL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        rd = model_cfg.get('RESOLUTION_DIV', [1.0])
        self.model_cfg.VFE.RESOLUTION_DIV = rd
        self.model_cfg.BACKBONE_3D.RESOLUTION_DIV = rd
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

        self.vfe, self.backbone_3d, self.backbone_2d, self.dense_head = self.module_list

        self.resolution_dividers = rd
        self.num_res = len(self.resolution_dividers)
        self.latest_losses = [0.] * self.num_res
        self.res_aware_1d_batch_norms, self.res_aware_2d_batch_norms = get_all_resawarebn(self)
        self.res_idx = 0

        self.alternate_order = self.model_cfg.get('ALTERNATE_ORDER', False)

    def forward_once(self, batch_dict):
        resdiv = self.resolution_dividers[self.res_idx]
        batch_dict['resolution_divider'] = resdiv
        self.vfe.adjust_voxel_size_wrt_resolution(self.res_idx)
        set_bn_resolution(self.res_aware_1d_batch_norms, self.res_idx)
        set_bn_resolution(self.res_aware_2d_batch_norms, self.res_idx)
        self.dense_head.adjust_voxel_size_wrt_resolution(resdiv)

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            return loss, tb_dict, disp_dict
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
            total_loss = 0.0
            tb_dict_combined = {}
            disp_dict_combined = {}

            if self.alternate_order:
                res_indices = range(self.num_res - 1, -1, -1)  # High to low
            else:
                res_indices = range(self.num_res)  # Low to high
            
            # Process all resolutions and accumulate losses without backpropagation
            for ridx in res_indices:
                self.res_idx = ridx
                
                # Forward pass for this resolution
                loss, curr_tb_dict, curr_disp_dict = self.forward_once(batch_dict)
                
                # Add to total loss (keeping gradients)
                total_loss = total_loss + loss
                
                # Store metrics with resolution prefix for logging
                res_prefix = f'res_{ridx}_'
                for k, v in curr_tb_dict.items():
                    tb_dict_combined[res_prefix + k] = v
                
                for k, v in curr_disp_dict.items():
                    disp_dict_combined[res_prefix + k] = v
            
            # Add the final loss to the logging dict
            tb_dict_combined['total_loss'] = total_loss.item()
            
            # Return the combined loss (with gradient information intact)
            ret_dict = {'loss': total_loss}
            return ret_dict, tb_dict_combined, disp_dict_combined
        else:
            # For inference, just use the first resolution
            self.res_idx = 0
            pred_dicts, recall_dicts = self.forward_once(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
            """
            Calculate training losses while preserving gradient information
            """
            disp_dict = {}
            
            # Get the loss tensor from the dense head
            loss_rpn, tb_dict_head = self.dense_head.get_loss()
            
            # Create a copy of the dictionary with .item() for logging
            tb_dict = {
                'loss_rpn': loss_rpn.item(),
            }
            
            # Add the head's metrics to our logging dict
            for k, v in tb_dict_head.items():
                if isinstance(v, torch.Tensor):
                    tb_dict[k] = v.item()  # Convert tensors to scalars for logging
                else:
                    tb_dict[k] = v
            
            # Return the loss tensor with gradients intact
            return loss_rpn, tb_dict, disp_dict

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
