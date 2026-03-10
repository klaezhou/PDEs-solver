import torch
import torch.optim as optim
from torch.optim import _functional as F
from viz.callbacks import Callback
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from train.proj import proj_step , Projection
from viz.rank_callback import RankCallback
class Trainer:
    """
    Trainer Module:
    ---------------
    1) Orchestrates the training loop using the injected model and equation.
    2) Polymorphism: calls eq.compute_loss(model, batch) regardless of PDE type.
    """
    def __init__(self, model, equation, args,callbacks=None):
        self.model = model
        self.eq = equation
        self.args = args

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # scheduler is optional
        self.use_scheduler = getattr(self.args, "use_scheduler", True)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=getattr(self.args, "lr_step_size", 5000),
            gamma=getattr(self.args, "lr_gamma", 0.7)
        )
        
        self.epochs = getattr(self.args, "iters", 10000)
        self.use_lbfgs = getattr(self.args, "use_lbfgs", False)
        self.lbfgs_iter = getattr(self.args, "lbfgs_iter", 500)
        self.lbfgs_max_iter = getattr(self.args, "lbfgs_max_iter", 50)
        self.lbfgs_lr = getattr(self.args, "lbfgs_lr", 1.0)
        
        # args for projection
        self.nx= getattr(args, "int_grid_n", 200)
        self.ny= getattr(args, "int_grid_n", 200)
        self.lowx= getattr(args, "int_domain_lowx", -1.0)
        self.highx= getattr(args, "int_domain_highx", 1.0)
        self.lowy= getattr(args, "int_domain_lowy", -1.0)
        self.highy= getattr(args, "int_domain_highy", 1.0)
        self.proj_g_upadate_freq = getattr(args, "proj_g_upadate_freq", 100)
        # 遍历 callbacks 列表，查找 RankCallback
        feature_getter = None
        for callback in callbacks:
            if isinstance(callback, RankCallback):  # 判断是否是 RankCallback 类型
                feature_getter = callback.feature_getter  # 获取 feature_getter
                break  # 找到后就可以退出循环

        self.feature_getter = feature_getter
        
        self.log_freq=None
        self.iter_base=0

        
        self.callbacks = callbacks or []
        self.l_history = []   # plot loss curve

    def _step_adam(self, data):
        """
        One training step using a FIXED batch 'data'.
        data: dict returned by eq.get_data(...)
        train the model for one step and return loss dict 
        """
        self.optimizer.zero_grad(set_to_none=True)

        loss_out = self.eq.compute_loss(self.model, data)

        # expect dict with "total", but keep it robust
        if isinstance(loss_out, dict):
            total_loss = loss_out["total"]
            loss_dict = loss_out
        else:
            total_loss = loss_out
            loss_dict = {"total": total_loss}

        total_loss.backward()
        self.optimizer.step()

        if self.use_scheduler and self.scheduler is not None:
            self.scheduler.step()

        # convert for logging (Tensor -> float)
        log_losses = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                log_losses[k] = float(v.detach().cpu())
            else:
                log_losses[k] = float(v)

        return log_losses
    
    def train_proj_adam(self, data):
        # 初始化参数的动量、平方梯度和更新步数
        # (1) train begin
        epochs=self.epochs
        self._set_phase("proj_adam")
        for cb in self.callbacks:
            cb.on_train_begin(self)
            
        self.optimizer_head= optim.Adam(self.model.head.parameters(), lr=self.args.lr)
            
        feature_params = list(self.model.feature.parameters()) 
        exp_avg = [torch.zeros_like(param) for param in feature_params]
        exp_avg_sq = [torch.zeros_like(param) for param in feature_params]
        max_exp_avg_sqs = [torch.zeros_like(param) for param in feature_params]
        state_steps = [torch.tensor(0.0, device=param.device) for param in feature_params]

        # 训练循环
        it=self.iter_base
        for epoch in range(epochs):  
            it += 1
            self.optimizer_head.zero_grad(set_to_none=True)
            
            loss_out = self.eq.compute_loss(self.model, data)
            if isinstance(loss_out, dict):
                total_loss = loss_out["total"]
                loss_dict = loss_out
            else:
                total_loss = loss_out
                loss_dict = {"total": total_loss}

            total_loss.backward()
            self.optimizer_head.step()
            # 获取梯度
            grads = [param.grad for param in feature_params]
            old_params = parameters_to_vector([p.detach().clone() for p in self.model.feature.parameters()])
            # 使用 F.adam 执行一步优化，并更新状态
            
            with torch.no_grad():
                for s in state_steps: s += 1 # 别忘了手动加步数
                F.adam(feature_params, grads, exp_avg, exp_avg_sq, max_exp_avg_sqs, state_steps, 
                    beta1=0.9, 
                    beta2=0.999, 
                    lr=self.args.lr, 
                    weight_decay=0, 
                    eps=1e-8, 
                    amsgrad=False, 
                    maximize=False
                )
            
            new_params = parameters_to_vector(feature_params)
            
            delta=new_params - old_params
            if epoch%self.args.proj_g_update_freq==0 or epoch ==0:
                for p in self.model.feature.parameters():
                    p.requires_grad_(True)
                
                delta_proj,alpha,g=proj_step(delta,self.model,self.feature_getter,self.nx,self.ny,self.lowx,self.highx,self.lowy,self.highy)
                print(f"[Projection] iter={it} alpha={alpha:.3e}")
            else:
                delta_proj,alpha=Projection(delta,g)
                # print(f"[Projection] iter={it} alpha={alpha:.3e}")
            
            vector_to_parameters(old_params + delta_proj, self.model.feature.parameters())
            
            self.model.zero_grad(set_to_none=True)
            
            log_losses = {}
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    log_losses[k] = float(v.detach().cpu())
                else:
                    log_losses[k] = float(v)
                    
            if it % self.log_freq == 0:
                self._print_log(it, log_losses)
            #(2) iter end
            for cb in self.callbacks:
                cb.on_iter_end(self, it, log_losses)
                
            
        self.iter_base += epochs
        # (3) train end
        for cb in self.callbacks:
            cb.on_train_end(self)


    def train_adam(self, data):
        " use adam to train for self.epochs "
        self.model.train()
        epochs=self.epochs
        # (1) train begin
        self._set_phase("adam")
        for cb in self.callbacks:
            cb.on_train_begin(self)
        
        it=self.iter_base
        for epoch in range(epochs):
             # losses is  log_losses which is a dict e.g. {"total":..., "pde":..., "bc":...}  defined in eq.compute_loss
            losses = self._step_adam(data)

            # epoch is 0-based; print at 1-based step
            it += 1
            if it % self.log_freq == 0:
                self._print_log(it, losses)
            # (2) iter end
            for cb in self.callbacks:
                cb.on_iter_end(self, it, losses)
                
        # (3) train end
        self.iter_base += epochs
        for cb in self.callbacks:
            cb.on_train_end(self)
            
            
    def train_lbfgs(self, data):
        """
        One LBFGS phase (usually after Adam).
        LBFGS uses full-batch + closure.
        """
        self.model.train()
        
        self._set_phase("lbfgs")
        
        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_lr,
            max_iter=self.lbfgs_max_iter,
            line_search_fn="strong_wolfe"
        )

        it_base = self.iter_base

        def closure():
            optimizer.zero_grad(set_to_none=True)

            loss_out = self.eq.compute_loss(self.model, data)
            if isinstance(loss_out, dict):
                total_loss = loss_out["total"]
            else:
                total_loss = loss_out

            total_loss.backward()
            return total_loss

        # LBFGS step will call closure multiple times internally
        for step in range(self.lbfgs_iter):
            step=step+1
            loss = optimizer.step(closure)

            # log once after LBFGS
        
            loss_out = self.eq.compute_loss(self.model, data)
            if isinstance(loss_out, dict):
                log_losses = {k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
                            for k, v in loss_out.items()}
            else:
                log_losses = {"total": float(loss_out.detach().cpu())}

            # trigger callbacks once (as "iter end")
            it = it_base +step
            if it % self.log_freq == 0:
                self._print_log(it, log_losses)
            for cb in self.callbacks:
                cb.on_iter_end(self, it, log_losses)           

        for cb in self.callbacks:
            cb.on_train_end(self)


        return log_losses
    
    

    def _print_log(self, it, losses):
        """print training log"""
        total = losses.get("total", None)
        if total is None:
            raise KeyError("loss dict must contain key 'total'")

        log_str = f"[Iter {it:06d}] Total: {total:.6e}"

        for k in sorted(losses.keys()):
            if k == "total":
                continue
            log_str += f" | {k.upper()}: {losses[k]:.6e}"

        lr = self.optimizer.param_groups[0]["lr"]
        log_str += f" | LR: {lr:.3e}"

        print(log_str)
    def _set_phase(self, phase: str):
        self.phase = phase
        for cb in self.callbacks:
            cb.on_phase_begin(self, phase)
            
        self.log_freq=self.args.log_freq.get(phase, 100)