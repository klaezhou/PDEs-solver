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
    parser.add_argument("--eq", type=str, default="poisson", help="Equation name (poisson, ac, etc.)")
    parser.add_argument("--device", type=str, default="cuda:6") #fb 6
    parser.add_argument("--model", type=str, default="mlp", help="Model architecture (moe_d, mlp , moe_d_w etc.)")
    parser.add_argument("--seed", type=int, default=2026)
    
    #mlp 
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--out_size", type=int, default=1)
    parser.add_argument("--mlp_hidden_size", type=int, default=30)
    parser.add_argument("--mlp_depth", type=int, default=3)
    
    
    
    parser.add_argument("--bump_depth", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=80)
    parser.add_argument("--center_mins", type=list, default=[-1.0, -1.0])
    parser.add_argument("--center_maxs", type=list, default=[1.0, 1.0])
    parser.add_argument("--center_steps", type=list, default=[0.1, 0.1])
    parser.add_argument("--radius",type=float,default=0.3)
    parser.add_argument("--anchor_weight", type=float, default=1000.0)
    # --- sampling ---
    parser.add_argument("--sample_method", type=str, default="grid") # random or grid
    parser.add_argument("--Nf", type=int, default=5000)
    parser.add_argument("--Nb", type=int, default=100) 
    parser.add_argument("--nx", type=int, default=40) #200
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--n_per_edge", type=int, default=100)
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bc", type=float, default=1.0)


    # --- modeling ---s
    parser.add_argument("--use_double", action="store_true", default=True)
    
    # --- training ---
    parser.add_argument("--adam_iters", type=int, default=50000)  # lm epochs
    parser.add_argument("--use_scheduler", type=bool, default=True)  #  use scheduler
    parser.add_argument("--lm_epochs", type=int, default=2000)  # lm epochs
    parser.add_argument("--lm_beta_train", type=bool, default=False) # lbfgs lr
    parser.add_argument("--adam_lr", type=float, default=2e-3) # adam lr
    parser.add_argument("--log_freq", type=dict, default={"adam": 500, "lbfgs": 20,"proj_adam":500,"lm":10}) # in trainer.py for print loss
    parser.add_argument("--checkpoint_freq", type=dict, default={"adam": 1000, "lbfgs": 1000,"proj_adam":10000,"lm":10000}) # in checkpoint_callback.py for saving model
    parser.add_argument("--rard_freq", type=dict, default={"adam":2000,"lbfgs":500,"lm":50}) 
    
    # --- plot ---
    parser.add_argument("--save_dir", type=str, default="/home/zhy/Zhou/DLPDEs/dlpdes/outputs/poisson")
    parser.add_argument("--plot_freq", type=dict, default={"adam": 500, "lbfgs": 50,"proj_adam":500,"lm":20} ) # in error_plot_callback.py for plot error
    parser.add_argument("--loss_freq", type=dict, default={"adam": 5, "lbfgs": 5,"proj_adam":5,"lm":5}) # in checkpoint_callback.py for saving model
    parser.add_argument("--eval_grid_n", type=int, default=150)
    
    # --- rank callback ---
    parser.add_argument("--rank_freq", type=dict, default={"adam": 100, "lbfgs": 30,"proj_adam":5000,"lm":20}) # in rank_callback.py for evaluating rank
    parser.add_argument("--int_grid_n", type=int, default=250)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--int_domain_lowx", type=float, default=-1.0)
    parser.add_argument("--int_domain_highx", type=float, default=1.0)
    parser.add_argument("--int_domain_lowy", type=float, default=-1.0)
    parser.add_argument("--int_domain_highy", type=float, default=1.0)
    
    
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
    feature_getter= get_feature_getter(args)
    rank_cb=    RankCallback(args=args, equation=eq, feature_getter=feature_getter, freq_dict=args.rank_freq)
    time_cb=    TimePlotCallback(args=args, freq_dict=args.loss_freq)
    resample_cb=ResamplePlotCallback(args=args,freq_dict=args.rard_freq)
    callbacks = [err_cb, loss_cb, check_cb, time_cb,rank_cb]
    # callbacks_new= [err_cb, loss_cb, time_cb]
    pipe = Pipeline(args=args, equation=eq, callbacks=callbacks)
    
    print(f"--- Starting {args.eq.upper()} train ---")
    


    # pipe.load_checkpoint("/home/zhy/Zhou/DLPDEs/dlpdes/outputs/dcei/_log_model/ckpt_iter_002000.pt")
    # pipe.trainer.model.freeze_all_parameters()
    # pipe.trainer.model.unfreeze_shared()
    # pipe.trainer.model._report_trainable()
    # pipe.trainer.train_adam(pipe.data)
    # pipe.trainer.train_lbfgs(pipe.data)
    
    # pipe.reset_model()  
    # pipe.reset_trainer()
    # pipe.trainer.train_lbfgs(pipe.data)
    # pipe.trainer.train_proj_adam(pipe.data)
   
    
    # pipe.trainer.train_lm(pipe.data)
    
    # print(f"--- {args.eq.upper()} lm train begin ---")
    
    # pipe.trainer.model.freeze_all_parameters()
    # 
    # pipe.trainer.model._report_trainable()
    pipe.trainer.train_lm(pipe.data)
    # model_old = copy.deepcopy(pipe.trainer.model)
    # pipe.trainer.train_slm_bump(pipe.data,model_old)
   
    
    # for i in range(1000):
    #     model_old = copy.deepcopy(pipe.trainer.model)
    #     train_center_idx = torch.randperm(pipe.trainer.model.num_centers)[:1].tolist()
    #     # train_center_idx=[i]
    #     print("selected centers:", train_center_idx)
        
    #     local_data = get_local_data_around_centers(
    #         pipe.eq,
    #         model=pipe.trainer.model,
    #         center_idx=train_center_idx,
    #         radius=0.5,            
    #         r_lmb=0.8,
    #         Nf=1000,     # 局部 interior 点数
    #         N_lm_b=1000,
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
    #     pipe.trainer.model.unfreeze_local_centers(train_center_idx)
    #     # pipe.trainer.model.unfreeze_shared()
    #     pipe.trainer.model._report_trainable()
    #     # pipe.trainer.train_lbfgs(local_data)
    #     pipe.trainer.train_slm_bump(local_data,model_old)
    #     # pipe.trainer.train_adam(local_data)

    
    
    print(f"--- {args.eq.upper()} train finished ---")
    

    


main()