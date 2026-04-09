import torch
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
    parser.add_argument("--eq", type=str, default="dcei", help="Equation name (poisson, ac, etc.)")
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--model", type=str, default="bump", help="Model architecture (moe_d, mlp , moe_d_w etc.)")
    parser.add_argument("--seed", type=int, default=2025)
    
    # --- sampling ---
    parser.add_argument("--sample_method", type=str, default="grid") # random or grid
    parser.add_argument("--Nf", type=int, default=5000)
    parser.add_argument("--Nb", type=int, default=100)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--n_per_edge", type=int, default=100)
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bc", type=float, default=20.0)


    # --- modeling ---s
    parser.add_argument("--use_double", action="store_true", default=False)
    
    # --- training ---
    parser.add_argument("--adam_iters", type=int, default=100000)  # lm epochs
    parser.add_argument("--use_scheduler", type=bool, default=True)  #  use scheduler
    parser.add_argument("--lm_epochs", type=int, default=2000)  # lm epochs
    parser.add_argument("--lm_beta_train", type=bool, default=False) # lbfgs lr
    parser.add_argument("--adam_lr", type=float, default=8e-2) # adam lr
    parser.add_argument("--log_freq", type=dict, default={"adam": 500, "lbfgs": 20,"proj_adam":500,"lm":20}) # in trainer.py for print loss
    parser.add_argument("--checkpoint_freq", type=dict, default={"adam": 10000, "lbfgs": 1000,"proj_adam":10000,"lm":10000}) # in checkpoint_callback.py for saving model

    
    # --- plot ---
    parser.add_argument("--save_dir", type=str, default="/home/zhy/Zhou/DLPDEs/dlpdes/outputs/dcei")
    parser.add_argument("--plot_freq", type=dict, default={"adam": 500, "lbfgs": 50,"proj_adam":500,"lm":100} ) # in error_plot_callback.py for plot error
    parser.add_argument("--loss_freq", type=dict, default={"adam": 5, "lbfgs": 5,"proj_adam":5,"lm":5}) # in checkpoint_callback.py for saving model
    parser.add_argument("--eval_grid_n", type=int, default=100)
    
    # --- rank callback ---
    parser.add_argument("--rank_freq", type=dict, default={"adam": 5000, "lbfgs": 30,"proj_adam":5000,"lm":200}) # in rank_callback.py for evaluating rank
    parser.add_argument("--int_grid_n", type=int, default=30)
    parser.add_argument("--int_domain_lowx", type=float, default=-1.0)
    parser.add_argument("--int_domain_highx", type=float, default=1.0)
    parser.add_argument("--int_domain_lowy", type=float, default=-1.0)
    parser.add_argument("--int_domain_highy", type=float, default=1.0)
    
    parser.add_argument("--eps", type=float, default=1e-3)
    
    return parser.parse_args()

def main():
    args = parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get the equation instance using the factory
    eq = get_equation(args)

    # Initialize the pipeline with the equation
    
    err_cb = ErrorPlotCallback(args=args, equation=eq,freq_dict=args.plot_freq)
    loss_cb=LossPlotCallback(args=args,freq_dict=args.loss_freq)
    check_cb = CheckpointCallback(args=args, freq_dict=args.checkpoint_freq)
    # feature_getter= get_feature_getter(args)
    # rank_cb=    RankCallback(args=args, equation=eq, feature_getter=feature_getter, freq_dict=args.rank_freq)
    time_cb=    TimePlotCallback(args=args, freq_dict=args.loss_freq)
    resample_cb=ResamplePlotCallback(args=args)
    callbacks = [err_cb, loss_cb, check_cb, time_cb,resample_cb]
    # callbacks_new= [err_cb, loss_cb, time_cb]
    pipe = Pipeline(args=args, equation=eq, callbacks=callbacks)
    
    print(f"--- Starting {args.eq.upper()} train ---")
    
    pipe.trainer.model.freeze_all_parameters()
    pipe.trainer.model.unfreeze_local_centers([0])
    print(f"111")
    pipe.trainer.model._report_trainable()
    print(dict(pipe.trainer.model.named_parameters()))
    # pipe.trainer.train_adam(pipe.data)
    # pipe.load_checkpoint("/home/zhy/Zhou/DLPDEs/dlpdes/outputs/dcei/_log_model/ckpt_iter_020000.pt")
    # pipe.trainer.train_lbfgs(pipe.data)
    pipe.trainer.train_lm(pipe.data)
    # pipe.reset_model()  
    # pipe.reset_trainer()
    # # pipe.trainer.train_lbfgs(pipe.data)
    # pipe.trainer.train_proj_adam(pipe.data)
   
    
    # pipe.trainer.train_lm(pipe.data)
    
    
#     args.lm_epochs=15
#     pipe=Pipeline(args=args, equation=eq, callbacks=callbacks_new)
#     


    
# `pipe.trainer.train_lm(pipe.data)` is a method call that trains the model using the
# Levenberg-Marquardt algorithm. This algorithm is commonly used for optimization problems,
# particularly in the context of solving nonlinear least squares problems. During training, the
# algorithm adjusts the model parameters to minimize the loss function based on the provided data
# (`pipe.data`) and the equation being solved.
    
#     delta1=pipe.trainer.param_delta
# # 
#     pipe2=Pipeline(args=args, equation=eq, callbacks=callbacks_new)
#     pipe2.load_checkpoint("/home/zhy/Zhou/DLPDEs/dlpdes/outputs/dcei/_log_model/ckpt_iter_005000.pt")
#     pipe2.newdata()
#     pipe2.trainer.train_lm(pipe2.data)
#     delta2=pipe2.trainer.param_delta

#     if "update" not in delta1 or "update" not in delta2:
#         raise ValueError("delta1 or delta2 does not contain key 'update'")

#     # n = 5  # 例如比较前5个

#     # delta1_t = torch.cat([x.reshape(-1) for x in delta1["update"][:n]])
#     # delta2_t = torch.cat([x.reshape(-1) for x in delta2["update"][:n]])
#     delta1_t = delta1["update"][0].reshape(-1)
#     delta2_t = delta2["update"][0].reshape(-1)

#     delta_diff = torch.mean(torch.abs(delta1_t - delta2_t))/torch.mean(torch.abs(delta1_t))
#     print("rel_l1:",delta_diff)
#     rel_l2 = torch.norm(delta1_t - delta2_t,p=2) / torch.norm(delta1_t,p=2)
#     print("rel_l2:",rel_l2)
#     cos_sim = torch.dot(delta1_t, delta2_t) / (
#     torch.norm(delta1_t) * torch.norm(delta2_t) + 1e-12)
#     print("cos_sim:",cos_sim)
#     norm_ratio = torch.norm(delta2_t) / (torch.norm(delta1_t) + 1e-12)
#     print("norm_ratio:",norm_ratio)
    
    
    
    print(f"--- {args.eq.upper()} train finished ---")
    


main()