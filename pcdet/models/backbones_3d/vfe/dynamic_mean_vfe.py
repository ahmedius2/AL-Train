import torch

from .vfe_template import VFETemplate

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate


class DynamicMeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        res_divs = model_cfg.get('RESOLUTION_DIV', [1.0])

        self.voxel_params = []
        for resdiv in res_divs:
            voxel_size_tmp = [vs * resdiv for vs in voxel_size[:2]]
            grid_size_tmp = [int(gs / resdiv) for gs in grid_size[:2]]
            self.voxel_params.append((
                    voxel_size_tmp[0], #voxel_x
                    voxel_size_tmp[1], #voxel_y
                    voxel_size[2], #voxel_z constant
                    voxel_size_tmp[0] / 2 + point_cloud_range[0], #x_offset
                    voxel_size_tmp[1] / 2 + point_cloud_range[1], #y_offset
                    voxel_size[2] / 2 + point_cloud_range[2], #z_offset
                    grid_size_tmp[0] * grid_size_tmp[1] * grid_size[2], #scale_xyz
                    grid_size_tmp[1] * grid_size[2], #scale_yz
                    grid_size[2], #scale_z
                    torch.tensor(grid_size_tmp + [grid_size[2]]).cuda(), # grid_size
                    torch.tensor(voxel_size_tmp + [voxel_size[2]]).cuda()
            ))

        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        self.set_params(0)

    # Allows switching between different pillar sizes
    def set_params(self, idx):
        self.voxel_x, self.voxel_y, self.voxel_z, \
                self.x_offset, self.y_offset, self.z_offset,  \
                self.scale_xyz, self.scale_yz, self.scale_z, \
                self.grid_size, self.voxel_size = self.voxel_params[idx]

    def adjust_voxel_size_wrt_resolution(self, res_idx):
        self.set_params(res_idx)

    def get_output_feature_dim(self):
        return self.num_point_features

    @torch.no_grad()
    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                points: (batch_idx, x, y, z, i, e)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        #batch_size = batch_dict['batch_size']
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)
        points = points[mask]
        point_coords = point_coords[mask]
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                        point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['pillar_features']  = batch_dict['voxel_features']
        batch_dict['voxel_coords'] = voxel_coords.contiguous()
        batch_dict['pillar_coords']  = batch_dict['voxel_coords']
        return batch_dict
