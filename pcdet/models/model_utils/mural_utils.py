from pcdet.ops.norm_funcs.res_aware_bnorm import ResAwareBatchNorm1d, ResAwareBatchNorm2d

def set_bn_resolution(resawarebns, res_idx):
    for rabn in resawarebns:
        rabn.setResIndex(res_idx)

def get_all_resawarebn(model):
    resaware1dbns, resaware2dbns = [], []
    for module in model.modules():
        if isinstance(module, ResAwareBatchNorm1d):
            resaware1dbns.append(module)
        elif isinstance(module, ResAwareBatchNorm2d):
            resaware2dbns.append(module)
    return resaware1dbns, resaware2dbns
