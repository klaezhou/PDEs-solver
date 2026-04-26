# model/factory.py
from model.mlp import MLP, mlp_penultimate_getter
from model.moe_d_w import MOE_dense_weight
from .moe_d import MOE_dense, moe_penultimate_getter
from .moe_d_w import moew_penultimate_getter
from .mlp_2 import MLP_2
from .bump_mlp import LocalBumpTanhNet
from .bump_mlp import bump_mlp_penultimate_getter
from .fb_mlp import fb_mlp_penultimate_getter
from .fb_mlp import FBTanhNet
from .bump_mlp_time import LocalBumpTanhNet_time
from .bump_mlp_time import bump_mlp_time_penultimate_getter
from .bump_smlp import LocalBumpTanhNet_Separate, bump_smlp_penultimate_getter
# from .burgers import BurgersEquation 

def get_model(args):
    """
    根据命令行参数 --eq 自动选择对应的方程类
    """
    mapping = {
        "moe_d": MOE_dense,
        "mlp": MLP,
        "moe_d_w": MOE_dense_weight,
        "mlp_2": MLP_2,
        "bump": LocalBumpTanhNet,
        "fb_mlp": FBTanhNet,
        "bump_time": LocalBumpTanhNet_time,
        "bump_s": LocalBumpTanhNet_Separate,
    }
    target_class = mapping.get(args.model.lower())
    if not target_class:
        raise ValueError(f"Model {args.model} is not defined in factory.")
    return target_class(args)

def get_feature_getter(args):
    """
    根据命令行参数 --model 自动选择对应的 feature getter 函数
    """
    mapping = {
        "moe_d": moe_penultimate_getter,
        "mlp": mlp_penultimate_getter,
        "moe_d_w": moew_penultimate_getter,
        "bump": bump_mlp_penultimate_getter,
        "fb_mlp": fb_mlp_penultimate_getter,
        "bump_time": bump_mlp_time_penultimate_getter,
        "bump_s": bump_smlp_penultimate_getter,
    }
    target_getter = mapping.get(args.model.lower())
    if not target_getter:
        raise ValueError(f"Feature getter for model {args.model} is not defined in factory.")
    return target_getter