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
from model.factory import get_feature_getter
def parse_args():
    parser = argparse.ArgumentParser(description="PINNs Lightweight Solver")
    
    # --- basic config ---
    parser.add_argument("--eq", type=str, default="dcei", help="Equation name (poisson, ac, etc.)")
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--model", type=str, default="mlp", help="Model architecture (moe_d, mlp , moe_d_w etc.)")
    parser.add_argument("--seed", type=int, default=2026)
    
    # --- sampling ---
    parser.add_argument("--sample_method", type=str, default="grid") # random or grid
    parser.add_argument("--Nf", type=int, default=5000)
    parser.add_argument("--Nb", type=int, default=100)
    parser.add_argument("--nx", type=int, default=30)
    parser.add_argument("--ny", type=int, default=30)
    parser.add_argument("--n_per_edge", type=int, default=100)
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bc", type=float, default=1.0)


    # --- modeling ---s
    parser.add_argument("--use_double", action="store_true", default=True)
    
    # --- training ---
    parser.add_argument("--lm_epochs", type=int, default=50)  # lm epochs
    parser.add_argument("--lm_beta_train", type=bool, default=False) # lbfgs lr
    parser.add_argument("--lbfgs_iter", type=int, default=700) # lbfgs iters
    parser.add_argument("--lbfgs_lr", type=float, default=1.0) # lbfgs lr
    parser.add_argument("--lr", type=float, default=8e-4) # adam lr
    parser.add_argument("--lr_step_size", type=int, default=2000)
    parser.add_argument("--lr_gamma", type=float, default=0.8) 
    parser.add_argument("--log_freq", type=dict, default={"adam": 500, "lbfgs": 10,"proj_adam":500,"lm":20}) # in trainer.py for print loss
    parser.add_argument("--checkpoint_freq", type=dict, default={"adam": 10000, "lbfgs": 200,"proj_adam":10000,"lm":10000}) # in checkpoint_callback.py for saving model

    
    # --- plot ---
    parser.add_argument("--save_dir", type=str, default="/home/zhy/Zhou/DLPDEs/dlpdes/outputs/dcei")
    parser.add_argument("--plot_freq", type=dict, default={"adam": 500, "lbfgs": 50,"proj_adam":500,"lm":200} ) # in error_plot_callback.py for plot error
    parser.add_argument("--loss_freq", type=dict, default={"adam": 5, "lbfgs": 1,"proj_adam":5,"lm":5}) # in checkpoint_callback.py for saving model
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
    feature_getter= get_feature_getter(args)
    # rank_cb=    RankCallback(args=args, equation=eq, feature_getter=feature_getter, freq_dict=args.rank_freq)
    time_cb=    TimePlotCallback(args=args, freq_dict=args.loss_freq)
    callbacks = [err_cb, loss_cb, check_cb, time_cb]
    pipe = Pipeline(args=args, equation=eq, callbacks=callbacks)
    
    print(f"--- Starting {args.eq.upper()} train ---")
    # pipe.trainer.train_adam(pipe.data)
    # pipe.reset_model()  
    # pipe.reset_trainer()
    # pipe.trainer.train_lbfgs(pipe.data)
    # pipe.trainer.train_proj_adam(pipe.data)
    # pipe.trainer.train_lm(pipe.data)
    # pipe.refresh_data()
    # pipe.trainer.train_lm(pipe.data)
    print(f"--- {args.eq.upper()} train finished ---")
    


main()