import copy
import torch
import math
import argparse
from Equation.factory import get_equation 
from Pipeline.pipeline import Pipeline
from cb.callbacks import Callback
from cb.error_plot_callback import ErrorPlotCallback
from cb.loss_plot_callback import LossPlotCallback
from cb.checkpoint_callback import CheckpointCallback
from cb.rank_callback import RankCallback
from cb.time_plot_callback import TimePlotCallback
from cb.resample_callbacks import ResamplePlotCallback
from model.factory import get_feature_getter

def parse_args():
    parser = argparse.ArgumentParser(description="PINNs Lightweight Solver")
    
    # --- basic config ---
    parser.add_argument("--eq", type=str, default="helmholtz", help="Equation name (poisson, ac, etc.)")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--model", type=str, default="bump", help="Model architecture (moe_d, mlp , moe_d_w etc.)")
    parser.add_argument("--seed", type=int, default=2026)
    
    parser.add_argument("--bump_depth", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=80)
    parser.add_argument("--center_step",type=float,default=0.1)
    parser.add_argument("--radius",type=float,default=0.3)
    parser.add_argument("--anchor_weight", type=float, default=2000.0)
    # --- sampling ---
    parser.add_argument("--sample_method", type=str, default="grid") # random or grid
    parser.add_argument("--Nf", type=int, default=5000)
    parser.add_argument("--Nb", type=int, default=100)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--n_per_edge", type=int, default=300)
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bc", type=float, default=500.0)



    # --- modeling ---s
    parser.add_argument("--use_double", action="store_true", default=True)
    
    # --- training ---
    parser.add_argument("--adam_iters", type=int, default=100000)  # lm epochs
    parser.add_argument("--use_scheduler", type=bool, default=True)  #  use scheduler
    parser.add_argument("--lm_epochs", type=int, default=2000)  # lm epochs
    parser.add_argument("--lm_beta_train", type=bool, default=False) # lbfgs lr
    parser.add_argument("--adam_lr", type=float, default=2e-2) # adam lr
    parser.add_argument("--log_freq", type=dict, default={"adam": 500, "lbfgs": 20,"proj_adam":500,"lm":10}) # in trainer.py for print loss
    parser.add_argument("--checkpoint_freq", type=dict, default={"adam": 1000, "lbfgs": 1000,"proj_adam":10000,"lm":10000}) # in checkpoint_callback.py for saving model
    parser.add_argument("--rard_freq", type=dict, default={"adam":500,"lbfgs":500,"lm":100}) 
    
    # --- plot ---
    parser.add_argument("--save_dir", type=str, default="/home/zhy/Zhou/DLPDEs/dlpdes/outputs/hel")
    parser.add_argument("--plot_freq", type=dict, default={"adam": 500, "lbfgs": 50,"proj_adam":500,"lm":50} ) # in error_plot_callback.py for plot error
    parser.add_argument("--loss_freq", type=dict, default={"adam": 5, "lbfgs": 5,"proj_adam":5,"lm":5}) # in checkpoint_callback.py for saving model
    parser.add_argument("--eval_grid_n", type=int, default=150)
    
    # --- rank callback ---
    parser.add_argument("--rank_freq", type=dict, default={"adam": 100, "lbfgs": 30,"proj_adam":5000,"lm":50}) # in rank_callback.py for evaluating rank
    parser.add_argument("--int_grid_n", type=int, default=200)
    parser.add_argument("--int_domain_lowx", type=float, default=-1.0)
    parser.add_argument("--int_domain_highx", type=float, default=1.0)
    parser.add_argument("--int_domain_lowy", type=float, default=-1.0)
    parser.add_argument("--int_domain_highy", type=float, default=1.0)
    
    
    return parser.parse_args()
import math
import torch


def _filter_points_in_disks(points, centers, radius):
    """
    points:  [Nb, 2]
    centers: [K, 2]
    return:  所有落在任意一个圆盘内的 points
    """
    diff = points[:, None, :] - centers[None, :, :]   # [Nb, K, 2]
    dist_sq = (diff ** 2).sum(dim=-1)                 # [Nb, K]
    mask = (dist_sq <= radius ** 2).any(dim=1)        # [Nb]
    return points[mask]


def _sample_points_in_disk(
    center,
    n,
    radius,
    low=-1.0,
    high=1.0,
    device=None,
    dtype=None,
):
    """
    在 2D 圆盘内均匀随机采样，并限制在 box [low, high]^2 内

    Args:
        center: shape [2]
        n: 采样点数
        radius: 圆盘半径
        low, high: 外部 box 约束范围
        device, dtype: 默认跟随 center

    Returns:
        Tensor, shape [n, 2]
    """
    if device is None:
        device = center.device
    if dtype is None:
        dtype = center.dtype

    pts = []
    remain = n

    while remain > 0:
        m = max(remain * 2, 64)

        theta = 2.0 * math.pi * torch.rand(m, device=device, dtype=dtype)
        rr = radius * torch.sqrt(torch.rand(m, device=device, dtype=dtype))

        dx = rr * torch.cos(theta)
        dy = rr * torch.sin(theta)

        cand = torch.stack(
            [center[0] + dx, center[1] + dy],
            dim=1,
        )  # [m, 2]

        mask = (
            (cand[:, 0] >= low)
            & (cand[:, 0] <= high)
            & (cand[:, 1] >= low)
            & (cand[:, 1] <= high)
        )
        cand = cand[mask]

        take = min(remain, cand.shape[0])
        if take > 0:
            pts.append(cand[:take])
            remain -= take

    return torch.cat(pts, dim=0)


def _sample_points_in_annulus(
    center,
    n,
    r_inner,
    r_outer,
    low=-1.0,
    high=1.0,
    device=None,
    dtype=None,
):
    """
    在 2D 圆环内均匀随机采样，并限制在 box [low, high]^2 内

    圆环区域:
        r_inner <= ||x - center|| <= r_outer

    Args:
        center: shape [2]
        n: 采样点数
        r_inner: 内半径
        r_outer: 外半径，要求 r_outer > r_inner
        low, high: 外部 box 约束范围
        device, dtype: 默认跟随 center

    Returns:
        Tensor, shape [n, 2]
    """
    if r_outer <= r_inner:
        raise ValueError(f"Require r_outer > r_inner, but got {r_outer} <= {r_inner}")

    if device is None:
        device = center.device
    if dtype is None:
        dtype = center.dtype

    pts = []
    remain = n

    while remain > 0:
        m = max(remain * 2, 64)

        theta = 2.0 * math.pi * torch.rand(m, device=device, dtype=dtype)

        # 按面积均匀采样圆环
        rr_sq = (r_outer ** 2 - r_inner ** 2) * torch.rand(m, device=device, dtype=dtype) + r_inner ** 2
        rr = torch.sqrt(rr_sq)

        dx = rr * torch.cos(theta)
        dy = rr * torch.sin(theta)

        cand = torch.stack(
            [center[0] + dx, center[1] + dy],
            dim=1,
        )  # [m, 2]

        mask = (
            (cand[:, 0] >= low)
            & (cand[:, 0] <= high)
            & (cand[:, 1] >= low)
            & (cand[:, 1] <= high)
        )
        cand = cand[mask]

        take = min(remain, cand.shape[0])
        if take > 0:
            pts.append(cand[:take])
            remain -= take

    return torch.cat(pts, dim=0)


def get_local_data_around_centers(
    eq,
    model,
    center_idx,
    radius,
    r_lmb,
    Nf=2000,
    N_lm_b=200,
    X_b_global=None,
    low=-1.0,
    high=1.0,
):
    """
    围绕若干 centers 生成局部数据：

    1) X_f:
       在半径 radius 的圆盘内采样

    2) LM_b:
       在圆环内采样
           radius <= ||x-center|| <= r_lmb

    3) X_b:
       从全局边界点 X_b_global 中筛选出落在局部圆盘内的点

    Args:
        eq: PDE/方程对象，需提供 eq.f(x), eq.g(x)
        model: 模型，需有 model.centers, shape [num_centers, 2]
        center_idx: list[int] 或 int
        radius: 局部 interior 圆盘半径
        r_lmb: LM_b 圆环的外半径，必须满足 r_lmb > radius
        Nf: 局部 interior 点总数
        N_lm_b: LM_b 点总数
        X_b_global: 全局边界点
        low, high: 全局 box 范围

    Returns:
        dict with keys:
            X_f, X_b, f_f, g_b, LM_b
    """
    if isinstance(center_idx, int):
        center_idx = [center_idx]

    if r_lmb <= radius:
        raise ValueError(f"Require r_lmb > radius, but got r_lmb={r_lmb}, radius={radius}")

    centers = model.centers[center_idx]   # [K, 2]

    K = len(center_idx)
    device = centers.device
    dtype = centers.dtype

    # -------------------------
    # 1) 局部 interior 点 X_f
    # -------------------------
    n_base = Nf // K
    n_rem = Nf % K

    X_f_list = []
    for j in range(K):
        nj = n_base + (1 if j < n_rem else 0)
        cf = _sample_points_in_disk(
            center=centers[j],
            n=nj,
            radius=radius,
            low=low,
            high=high,
            device=device,
            dtype=dtype,
        )
        X_f_list.append(cf)

    X_f = torch.cat(X_f_list, dim=0).requires_grad_(True)

    # -------------------------
    # 2) 人工边界/过渡区域点 LM_b
    #    在圆环内采样
    # -------------------------
    lm_base = N_lm_b // K
    lm_rem = N_lm_b % K

    LM_b_list = []
    for j in range(K):
        nj = lm_base + (1 if j < lm_rem else 0)
        cb = _sample_points_in_annulus(
            center=centers[j],
            n=nj,
            r_inner=radius,
            r_outer=r_lmb,
            low=low,
            high=high,
            device=device,
            dtype=dtype,
        )
        LM_b_list.append(cb)

    LM_b = torch.cat(LM_b_list, dim=0).requires_grad_(True)

    # -------------------------
    # 3) 全局物理边界 X_b
    #    只保留落在局部圆盘内的边界点
    # -------------------------
    if X_b_global is None:
        raise ValueError("X_b_global is None. 先传入全局边界点。")

    X_b_candidate = X_b_global.detach().clone().to(device=device, dtype=dtype)

    X_b_local = _filter_points_in_disks(
        points=X_b_candidate,
        centers=centers,
        radius=radius,
    )

    X_b = X_b_local.requires_grad_(True)

    # -------------------------
    # 4) 真值/右端项
    # -------------------------
    f_f = eq.f(X_f)
    g_b = eq.g(X_b)

    return {
        "X_f": X_f,      # 局部 interior 点
        "X_b": X_b,      # 落在局部圆盘内的全局物理边界点
        "f_f": f_f,
        "g_b": g_b,
        "LM_b": LM_b,    # 圆环区域点
    }
def main():
    args = parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Get the equation instance using the factory
    eq = get_equation(args)
    # Initialize the pipeline with the equation
    err_cb = ErrorPlotCallback(args=args, equation=eq,freq_dict=args.plot_freq)
    loss_cb=LossPlotCallback(args=args,freq_dict=args.loss_freq)
    check_cb = CheckpointCallback(args=args, freq_dict=args.checkpoint_freq)
    feature_getter= get_feature_getter(args)
    rank_cb=    RankCallback(args=args, equation=eq, feature_getter=feature_getter, freq_dict=args.rank_freq)
    time_cb=    TimePlotCallback(args=args, freq_dict=args.loss_freq)
    resample_cb=ResamplePlotCallback(args=args,freq_dict=args.rard_freq)
    callbacks = [err_cb, loss_cb, check_cb, time_cb,rank_cb]
    # callbacks_new= [err_cb, loss_cb, time_cb]
    pipe = Pipeline(args=args, equation=eq, callbacks=callbacks)
    
    print(f"--- Starting {args.eq.upper()} train ---")
    
    # pipe.trainer.model.freeze_all_parameters()
    # pipe.trainer.model.unfreeze_local_centers([0])
    # pipe.trainer.model._report_trainable()

    # pipe.load_checkpoint("/home/zhy/Zhou/DLPDEs/dlpdes/outputs/hel/_log_model/ckpt_iter_002000.pt")
    # pipe.trainer.model.freeze_all_parameters()
    # pipe.trainer.model.unfreeze_shared()
    # pipe.trainer.model._report_trainable()
    pipe.trainer.train_adam(pipe.data)
    # pipe.trainer.train_lbfgs(pipe.data)
    
    # pipe.reset_model()  
    # pipe.reset_trainer()
    # pipe.trainer.train_lbfgs(pipe.data)
    # pipe.trainer.train_proj_adam(pipe.data)
   
    
    # pipe.trainer.train_lm(pipe.data)
    
    # print(f"--- {args.eq.upper()} lm train begin ---")
    
    # pipe.trainer.model.freeze_all_parameters()
    # pipe.trainer.model.unfreeze_shared()
    # pipe.trainer.model._report_trainable()
    # # pipe.trainer.train_lm(pipe.data)
    # model_old = copy.deepcopy(pipe.trainer.model)
    # pipe.trainer.train_slm_bump(pipe.data,model_old)
   
    
    # for i in range(1000):
    #     model_old = copy.deepcopy(pipe.trainer.model)
    #     # train_center_idx = torch.randperm(pipe.trainer.model.num_centers)[:1].tolist()
    #     train_center_idx=[i]
    #     print("selected centers:", train_center_idx)
        
    #     local_data = get_local_data_around_centers(
    #         pipe.eq,
    #         model=pipe.trainer.model,
    #         center_idx=train_center_idx,
    #         radius=0.2,            
    #         r_lmb=0.3,
    #         Nf=100,     # 局部 interior 点数
    #         N_lm_b=100,
    #         X_b_global=pipe.data["X_b"],  # 复用全局边界
    #         low=-1.0,
    #         high=1.0,
    #     )
        
    #     # local_data =filter_local_data_around_centers(pipe.data, pipe.trainer.model, train_center_idx, 0.3)
        
    #     print("=== local_data ===")
    #     for k, v in local_data.items():
    #         if torch.is_tensor(v):
    #             print(f"{k}: shape={tuple(v.shape)}, numel={v.numel()}")
    #         else:
    #             print(f"{k}: type={type(v)}")
                
    #     selected_centers = pipe.trainer.model.centers[train_center_idx].detach().cpu()
    #     print("selected centers pos:")
    #     for idx, c in zip(train_center_idx, selected_centers):
    #         print(f" center[{idx}] = {c.tolist()}")
    #     args.lm_epochs=20
    #     pipe.trainer.model.freeze_all_parameters()
    #     # pipe.trainer.model.unfreeze_local_centers(train_center_idx)
    #     pipe.trainer.model.unfreeze_shared()
    #     pipe.trainer.model._report_trainable()
    #     # pipe.trainer.train_lbfgs(local_data)
    #     pipe.trainer.train_slm_bump(local_data,model_old)
    #     # pipe.trainer.train_adam(local_data)

    
    
    print(f"--- {args.eq.upper()} train finished ---")
    

    


main()