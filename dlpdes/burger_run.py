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
    parser.add_argument("--eq", type=str, default="burgers1d")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--model", type=str, default="bump_time")
    parser.add_argument("--seed", type=int, default=2026)

    # --- Burgers parameter ---
    parser.add_argument("--nu", type=float, default=0.01 / math.pi)
    parser.add_argument(
        "--burgers_data_path",
        type=str,
        default="/home/zhy/Zhou/DLPDEs/dlpdes/data/ref/burgers_shock_mu_01_pi.mat"
    )

    # --- model config ---
    parser.add_argument("--hidden_dim", type=int, default=80)
    parser.add_argument("--mlp_depth", type=int, default=4)

    # 如果你的模型还需要这些参数，就保留，避免 factory 报错
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--bump_depth", type=int, default=1)

    parser.add_argument("--center_mins", type=list, default=[-1.0])
    parser.add_argument("--center_maxs", type=list, default=[1.0])
    parser.add_argument("--center_steps", type=list, default=[0.1])
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--anchor_weight", type=float, default=1000.0)

    # --- sampling ---
    parser.add_argument("--sample_method", type=str, default="grid")
    parser.add_argument("--Nf", type=int, default=10000)
    parser.add_argument("--Ni", type=int, default=200)
    parser.add_argument("--Nb", type=int, default=200)

    # grid mode / eval config
    parser.add_argument("--nx", type=int, default=200) #sampling
    parser.add_argument("--nt", type=int, default=200)
    parser.add_argument("--eval_nx", type=int, default=101) #error ,groud truth
    parser.add_argument("--eval_nt", type=int, default=11)

    # --- loss weights ---
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_ic", type=float, default=200.0)
    parser.add_argument("--w_bc", type=float, default=1.0)

    # --- precision ---
    parser.add_argument("--use_double", action="store_true", default=True)

    # --- training ---
    parser.add_argument("--adam_iters", type=int, default=50000)
    parser.add_argument("--adam_lr", type=float, default=5e-2)
    parser.add_argument("--use_scheduler", type=bool, default=True)

    parser.add_argument("--lm_epochs", type=int, default=2000)
    parser.add_argument("--lm_beta_train", type=bool, default=False)

    parser.add_argument(
        "--log_freq",
        type=dict,
        default={"adam": 500, "lbfgs": 20, "proj_adam": 500, "lm": 10}
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=dict,
        default={"adam": 1000, "lbfgs": 1000, "proj_adam": 10000, "lm": 10000}
    )
    parser.add_argument(
        "--rard_freq",
        type=dict,
        default={"adam": 2000, "lbfgs": 500, "lm": 50}
    )

    # --- plot ---
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/zhy/Zhou/DLPDEs/dlpdes/outputs/burgers1d"
    )
    parser.add_argument(
        "--plot_freq",
        type=dict,
        default={"adam": 500, "lbfgs": 50, "proj_adam": 500, "lm": 50}
    )
    parser.add_argument(
        "--loss_freq",
        type=dict,
        default={"adam": 5, "lbfgs": 5, "proj_adam": 5, "lm": 5}
    )


    # --- rank callback ---
    parser.add_argument(
        "--rank_freq",
        type=dict,
        default={"adam": 1000, "lbfgs": 30, "proj_adam": 5000, "lm": 50}
    )
    parser.add_argument("--int_grid_n", type=int, default=250)
    parser.add_argument("--int_domain_lowx", type=float, default=-1.0)
    parser.add_argument("--int_domain_highx", type=float, default=1.0)
    parser.add_argument("--int_domain_lowy", type=float, default=0.0)
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
    callbacks = [err_cb, loss_cb, check_cb, time_cb]
    # callbacks_new= [err_cb, loss_cb, time_cb]
    pipe = Pipeline(args=args, equation=eq, callbacks=callbacks)
    
    print(f"--- Starting {args.eq.upper()} train ---")
    


    # pipe.load_checkpoint("/home/zhy/Zhou/DLPDEs/dlpdes/outputs/dcei/_log_model/ckpt_iter_002000.pt")
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
    # 
    # pipe.trainer.model._report_trainable()
    # # pipe.trainer.train_lm(pipe.data)
    # model_old = copy.deepcopy(pipe.trainer.model)
    # pipe.trainer.train_slm_bump(pipe.data,model_old)
   
    

    
    
    print(f"--- {args.eq.upper()} train finished ---")
    

    


main()