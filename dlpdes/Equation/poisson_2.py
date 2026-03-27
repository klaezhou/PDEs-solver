# Equation/poisson.py
import numpy as np
import torch
import os
from torch.func import functional_call, jacrev, hessian, vmap
import torch
from Equation._base import BaseEquation
import math
import matplotlib.pyplot as plt
from torch.func import jacrev, vmap
class PoissonEquation_2(BaseEquation):
    """
    Poisson equation -Δu = f, on domain Ω , with Dirichler boundary condtion u=g at ∂Ω. 
    two dimensional case.
    """
    def f(self, x):
        """
        right term f(x)
        x: [N, dim]
        return: [N, 1]
        """
        # example: f = (10*.pi**2)*sin(3*pi*x1)*sin(1*pi*x2)

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        return (2*math.pi**2) * torch.sin(1*math.pi * x1) * torch.sin(1*math.pi * x2)

    def g(self, x):
        """
        boundary value g(x). u=g at boundary
        x: [N, dim]
        return: [N, 1]
        """
        # example: u=0 at boundary
        return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    
    
    def gradientu_autograd(self, model_fn, x):
        """
        model_fn: callable, x -> (u, q)
            u: [N,1]
            q: [N,dim]
        x: [N, dim]
        return: grad_u [N, dim]
        """
        x = x.clone().detach().requires_grad_(True)

        u, _ = model_fn(x)   # u: [N,1]

        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]   # [N, dim]

        return grad_u
    def divq_autograd(self, model_fn, x):
        """
        model_fn: callable, x -> (u, q)
            u: [N,1]
            q: [N,2]
        x: [N,2]
        return: div_q [N,1]
        """
        x = x.clone().detach().requires_grad_(True)

        _, q = model_fn(x)   # q: [N,2]

        q1 = q[:, 0:1]       # [N,1]
        q2 = q[:, 1:2]       # [N,1]

        dq1_dx = torch.autograd.grad(
            outputs=q1,
            inputs=x,
            grad_outputs=torch.ones_like(q1),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]   # [N,2]

        dq2_dx = torch.autograd.grad(
            outputs=q2,
            inputs=x,
            grad_outputs=torch.ones_like(q2),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]   # [N,2]

        div_q = dq1_dx[:, 0:1] + dq2_dx[:, 1:2]   # [N,1]
        return div_q
    def gradientu_jacrev(self, model_fn, x):
        """
        model_fn: callable, x -> [N,1]
        x: [N, dim]
        return: [N,1]
        """
        def scalar_u(x_single):
            # x_single: [dim]
            u,q = model_fn(x_single.unsqueeze(0))   
            return u.squeeze()                   
        
        def residual_single(x_single):
            grad_u = jacrev(scalar_u)(x_single)      # [dim]
            return grad_u

        grad = vmap(residual_single)(x)                 # [N]
        return grad                 # [N,1]
    
    def divq_jacrev(self, model_fn, x):
        """
        model_fn: callable, x -> [N,1]
        x: [N, dim]
        return: [N,1]
        """
        # def vector_q(x_single):
        #     u, q = model_fn(x_single.unsqueeze(0))
        #     return q.squeeze(0)              # [2]
        def q1(x_single):
            _, q = model_fn(x_single.unsqueeze(0))
            return q.squeeze(0)[0]

        def q2(x_single):
            _, q = model_fn(x_single.unsqueeze(0))
            return q.squeeze(0)[1]

        def lap_single(x_single):
            dq1_dx = jacrev(q1)(x_single)   # [2]
            dq2_dx = jacrev(q2)(x_single)   # [2]
            div_q = dq1_dx[0] + dq2_dx[1]
            return div_q

        div = vmap(lap_single)(x)                 # [N]
        return div.unsqueeze(1)                   # [N,1]

    def compute_loss(self, model, batch: dict, mode="jacrev"):
        x_f = batch["X_f"]

        if mode == "backward":
            divq = self.divq_autograd(model, x_f)
            gradu = self.gradientu_autograd(model, x_f)
        elif mode == "jacrev":
            divq = self.divq_jacrev(model, x_f)         # expected [N] or [N,1]
            gradu = self.gradientu_jacrev(model, x_f)   # expected [N,2]

        f_f = batch.get("f_f", self.f(x_f))             # expected [N] or [N,1]
        _, q = model(x_f)                               # expected [N,2]

        # ---- shape alignment ----
        if divq.ndim == 1:
            divq = divq.unsqueeze(1)    # [N,1]
        if f_f.ndim == 1:
            f_f = f_f.unsqueeze(1)      # [N,1]

        r_f1 = divq + f_f               # [N,1]
        r_f2 = gradu - q                # [N,2]

        loss_pde = 0.5 * torch.mean(r_f1**2) + 0.5 * torch.mean(r_f2**2)

        x_b = batch["X_b"]
        u_b, _ = model(x_b)
        g_b = batch.get("g_b", self.g(x_b))

        if u_b.ndim == 1:
            u_b = u_b.unsqueeze(1)
        if g_b.ndim == 1:
            g_b = g_b.unsqueeze(1)

        r_b = u_b - g_b                 # [Nb,1]
        loss_bc = 0.5 * torch.mean(r_b**2)

        w_pde = getattr(self.args, "w_pde", 1.0)
        w_bc  = getattr(self.args, "w_bc", 1.0)

        total_loss = w_pde * loss_pde + w_bc * loss_bc
        # print("mean |r_f1|^2 =", torch.mean(r_f1**2).item())
        # print("mean |r_f2|^2 =", torch.mean(r_f2**2).item())
        # print("mean |divq| =", torch.mean(torch.abs(divq)).item())
        # print("mean |gradu| =", torch.mean(torch.abs(gradu)).item())
        # print("mean |q| =", torch.mean(torch.abs(q)).item())

        # print("r_f1 shape:", r_f1.shape)
        # print("r_f2 shape:", r_f2.shape)
        # print("r_b shape:", r_b.shape)

        r = torch.cat([
            r_f1.reshape(-1),
            r_f2.reshape(-1),
            r_b.reshape(-1)
        ], dim=0)
        r1=r_f1
        r2= torch.cat([
            r_f2.reshape(-1),
            r_b.reshape(-1)
        ], dim=0)
        

        # print("all residual shape:", r.shape)

        r = r / math.sqrt(r.numel())

        loss_dict = {
            "loss": {
                "total": total_loss,
                "pde": loss_pde.detach(),
                "bc": loss_bc.detach(),
            },
            "residuals": {
                "all": r,
                # "r1": r1,
                # "r2": r2
            }
        }
        return loss_dict
    
    def get_data(self, data_loader):
        # 1) decide how many points to sample
        Nf = getattr(self.args, "Nf", 10000)
        Nb = getattr(self.args, "Nb", 2000)

        # 2) sample interior/boundary points using DataLoader tools
        sample_method = getattr(self.args, "sample_method", "grid") # "random" or "grid"
        #square domain [-1,1]x[-1,1] 
        if sample_method == "random":
            X_f = data_loader.sample_interior_box(Nf, dim=2, low=-1.0, high=1.0)
            X_b = data_loader.sample_boundary_box_2d(Nb, low=-1.0, high=1.0)
        elif sample_method == "grid":
            nx = getattr(self.args, "nx", 100)
            ny = getattr(self.args, "ny", 100)
            n_per_edge = getattr(self.args, "n_per_edge", 100)
            X_f = data_loader.sample_interior_grid_2d(nx=nx, ny=ny, low=-1.0, high=1.0, exclude_boundary=True)
            X_b = data_loader.sample_boundary_grid_2d(n_per_edge=n_per_edge, low=-1.0, high=1.0, include_corners=True)
        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")

        # 3) (recommended) precompute f and g values
        X_f = X_f.requires_grad_(True)
        X_b = X_b.requires_grad_(True)
        f_f = self.f(X_f)
        g_b = self.g(X_b)

        # 4) return a unified batch dict
        return {"X_f": X_f, "X_b": X_b, "f_f": f_f, "g_b": g_b}
    
    def exact_solution(self, x):
            """return exact solution at x for error analysis"""
            return torch.sin(1*torch.pi * x[:, 0:1]) * torch.sin(1*torch.pi * x[:, 1:2])
        
    @torch.no_grad()
    def plot_error(self, model, it: int, save_dir: str):
        """
        Save 2D absolute error heatmap and log L2 error.
        
        Creates/append to: save_dir/error_log.csv
        """
        os.makedirs(save_dir, exist_ok=True)

        # --- config ---
        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = self.args.device
        eps = 1e-12

        # --- build & cache grid (avoid rebuilding every time) ---
        cache_key = (grid_n, low, high, str(device))
        if not hasattr(self, "_eval_cache") or self._eval_cache.get("key") != cache_key:
            xs = torch.linspace(low, high, steps=grid_n, device=device)
            ys = torch.linspace(low, high, steps=grid_n, device=device)
            X, Y = torch.meshgrid(xs, ys, indexing="ij")
            grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [N,2]
            self._eval_cache = {
                "key": cache_key,
                "X_cpu": X.detach().cpu(),
                "Y_cpu": Y.detach().cpu(),
                "grid_xy": grid_xy,
            }

        X_cpu = self._eval_cache["X_cpu"]
        Y_cpu = self._eval_cache["Y_cpu"]
        grid_xy = self._eval_cache["grid_xy"]

        # --- evaluate ---
        model_was_training = model.training
        model.eval()
        pred,_ = model(grid_xy)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        exact = self.exact_solution(grid_xy)
        err = pred - exact  # [N,1]

        # L2 error (absolute & relative)
        l2_abs = torch.sqrt(torch.mean(err**2)).item()
        denom = torch.sqrt(torch.mean(exact**2)).item()
        l2_rel = l2_abs / (denom + eps)

        # reshape for plotting
        err_abs_grid = err.abs().reshape(grid_n, grid_n).detach().cpu()

        # --- plot & save image ---
        plt.figure()
        plt.title(f"|u_pred - u_exact| (iter={it})\nL2_abs={l2_abs:.3e}, L2_rel={l2_rel:.3e}")
        plt.pcolormesh(X_cpu, Y_cpu, err_abs_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        img_path = os.path.join(save_dir, f"poisson_error_iter_{it:06d}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        # --- append to csv log ---
        csv_path = os.path.join(save_dir, "poisson_error_log.csv")
        need_header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("iter,l2_abs,l2_rel\n")
            f.write(f"{it},{l2_abs:.12e},{l2_rel:.12e}\n")
            

        # --- console output (lightweight) ---
        print(f"[ErrorPlot] iter={it} | L2_abs={l2_abs:.3e} | L2_rel={l2_rel:.3e} | saved: {img_path}")

        if model_was_training:
            model.train()
            
    def plot_ground_truth(self,  save_dir):
        """
        Plot and save the exact solution u_exact on a 2D grid.
        Note: model is not used here, kept for a unified interface.
        """
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = self.args.device

        # build grid (cache to avoid rebuilding every call)
        cache_key = (grid_n, low, high, str(device))
        if not hasattr(self, "_gt_cache") or self._gt_cache.get("key") != cache_key:
            xs = torch.linspace(low, high, steps=grid_n, device=device)
            ys = torch.linspace(low, high, steps=grid_n, device=device)
            X, Y = torch.meshgrid(xs, ys, indexing="ij")
            grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [N,2]
            self._gt_cache = {
                "key": cache_key,
                "X_cpu": X.detach().cpu(),
                "Y_cpu": Y.detach().cpu(),
                "grid_xy": grid_xy,
            }

        X_cpu = self._gt_cache["X_cpu"]
        Y_cpu = self._gt_cache["Y_cpu"]
        grid_xy = self._gt_cache["grid_xy"]

        # exact solution on grid
        exact = self.exact_solution(grid_xy)
        exact_grid = exact.reshape(grid_n, grid_n).detach().cpu()

        # plot & save
        plt.figure()
        plt.title("u_exact (ground truth)")
        plt.pcolormesh(X_cpu, Y_cpu, exact_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        out_path = os.path.join(save_dir, "possion_ground_truth.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
    
    @torch.no_grad()
    def plot_u(self, model,  save_dir: str):
        """
        Save 2D absolute error heatmap and log L2 error.
        
        Creates/append to: save_dir/error_log.csv
        """
        os.makedirs(save_dir, exist_ok=True)

        # --- config ---
        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = self.args.device
        eps = 1e-12

        # --- build & cache grid (avoid rebuilding every time) ---
        cache_key = (grid_n, low, high, str(device))
        if not hasattr(self, "_eval_cache") or self._eval_cache.get("key") != cache_key:
            xs = torch.linspace(low, high, steps=grid_n, device=device)
            ys = torch.linspace(low, high, steps=grid_n, device=device)
            X, Y = torch.meshgrid(xs, ys, indexing="ij")
            grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [N,2]
            self._eval_cache = {
                "key": cache_key,
                "X_cpu": X.detach().cpu(),
                "Y_cpu": Y.detach().cpu(),
                "grid_xy": grid_xy,
            }

        X_cpu = self._eval_cache["X_cpu"]
        Y_cpu = self._eval_cache["Y_cpu"]
        grid_xy = self._eval_cache["grid_xy"]

        # --- evaluate ---
        model_was_training = model.training
        model.eval()

        pred ,_= model(grid_xy)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)



        # reshape for plotting
        u_grid = pred.reshape(grid_n, grid_n).detach().cpu()

        # --- plot & save image ---
        plt.figure()
        plt.title(f"u_pred")
        plt.pcolormesh(X_cpu, Y_cpu, u_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        img_path = os.path.join(save_dir, f"poisson_predict.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        if model_was_training:
            model.train()
        
        

    

