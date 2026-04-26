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
    parser = argparse.ArgumentParser(description="PINNs Lightweight Solver for 10D")
    
    # --- basic config ---
    parser.add_argument("--eq", type=str, default="poisson10d", help="Equation name (poisson, ac, etc.)")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--model", type=str, default="bump", help="Model architecture (moe_d, mlp , moe_d_w etc.)")
    parser.add_argument("--seed", type=int, default=2026)
    
    parser.add_argument("--bump_depth", type=int, default=1)
    parser.add_argument("--input_dim", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=20)
    parser.add_argument("--input_size", type=int, default=5)
    parser.add_argument("--center_step",type=float,default=1.0)
    parser.add_argument("--radius",type=float,default=2.0)
    
    
    # --- sampling (CRITICAL FOR 10D) ---
    # 10维空间下网格采样会内存爆炸，必须使用 random
    parser.add_argument("--sample_method", type=str, default="random") 
    # 增加内部采样点以覆盖10维空间
    parser.add_argument("--Nf", type=int, default=40000) 
    # 增加边界采样点 (10维的边界非常庞大)
    parser.add_argument("--Nb", type=int, default=5000) 
    
    # 既然是 random 采样，nx, ny 等网格参数在此次训练中将不再生效，保留仅为了兼容其他方程
    parser.add_argument("--nx", type=int, default=30)
    parser.add_argument("--ny", type=int, default=30)
    parser.add_argument("--n_per_edge", type=int, default=100)
    
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bc", type=float, default=1.0)

    # --- modeling ---
    parser.add_argument("--use_double", action="store_true", default=True)
    
    # --- training ---
    parser.add_argument("--adam_iters", type=int, default=50000)  # lm epochs
    parser.add_argument("--lm_epochs", type=int, default=50)  # lm epochs
    parser.add_argument("--lm_beta_train", type=bool, default=False) 
    parser.add_argument("--lbfgs_iter", type=int, default=700) # lbfgs iters
    parser.add_argument("--lbfgs_lr", type=float, default=1.0) # lbfgs lr
    parser.add_argument("--lr", type=float, default=1e-2) # adam lr
    parser.add_argument("--lr_step_size", type=int, default=2000)
    parser.add_argument("--lr_gamma", type=float, default=0.8) 
    parser.add_argument("--log_freq", type=dict, default={"adam": 500, "lbfgs": 10,"proj_adam":500,"lm":20}) 
    parser.add_argument("--checkpoint_freq", type=dict, default={"adam": 10000, "lbfgs": 200,"proj_adam":10000,"lm":10000}) 

    # --- plot ---
    parser.add_argument("--save_dir", type=str, default="/home/zhy/Zhou/DLPDEs/dlpdes/outputs/ps_10D")
    parser.add_argument("--plot_freq", type=dict, default={"adam": 500, "lbfgs": 50,"proj_adam":500,"lm":200} ) 
    parser.add_argument("--loss_freq", type=dict, default={"adam": 5, "lbfgs": 1,"proj_adam":5,"lm":5}) 
    
    # 画图切片的网格分辨率，100x100 的 2D 切片是没问题的
    parser.add_argument("--eval_grid_n", type=int, default=100)
    
    # --- rank callback ---
    # 警告：如果在 10D 开启 rank_callback，请确保其内部不使用全网格积分，否则会 OOM
    parser.add_argument("--rank_freq", type=dict, default={"adam": 5000, "lbfgs": 30,"proj_adam":5000,"lm":200}) 
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

    # Initialize the callbacks
    err_cb = ErrorPlotCallback(args=args, equation=eq, freq_dict=args.plot_freq)
    loss_cb = LossPlotCallback(args=args, freq_dict=args.loss_freq)
    check_cb = CheckpointCallback(args=args, freq_dict=args.checkpoint_freq)
    
    # rank_cb = RankCallback(args=args, equation=eq, feature_getter=feature_getter, freq_dict=args.rank_freq)
    time_cb = TimePlotCallback(args=args, freq_dict=args.loss_freq)
    # resample_cb=ResamplePlotCallback(args=args)
    callbacks = [err_cb, loss_cb, check_cb, time_cb]
    
    # Initialize the pipeline
    pipe = Pipeline(args=args, equation=eq, callbacks=callbacks)
    
    print(f"--- Starting {args.eq.upper()} train in {eq.dim if hasattr(eq, 'dim') else 'Unknown'}D Space ---")
    
    # 推荐从 Adam 开始训练高维问题，再用 LBFGS 收尾
    pipe.trainer.train_adam(pipe.data)
    # pipe.reset_model()  
    # pipe.reset_trainer()
    # pipe.trainer.train_lbfgs(pipe.data)
    
    # pipe.trainer.train_proj_adam(pipe.data)
    # pipe.trainer.train_lm(pipe.data)
    # pipe.refresh_data()
    # pipe.trainer.train_lm(pipe.data)
    print(f"--- {args.eq.upper()} train finished ---")

if __name__ == "__main__":
    main()