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
class PoissonEquation(BaseEquation):
    """
    Poisson equation -Δu = f, on domain Ω , with Dirichler boundary condtion u=g at ∂Ω. 
    two dimensional case.
    """
    def f(self, x):
        """
        right term f(x) for -Laplace u = f
        x: [N, 2]
        return: [N, 1]
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        # parameters
        A = 0.0
        k = 20
        K = k * math.pi

        cx, cy = 0.0, 0.0
        R = 0.20
        
        p1,p2=2,2

        # base term: -Δ[sin(pi x1) sin(pi x2)]
        f0 = ( p1**2+p2**2)*math.pi**2 * torch.sin(p1*math.pi * x1) * torch.sin(p2*math.pi * x2)

        # localized bump
        q = ((x1 - cx) / R) ** 2 + ((x2 - cy) / R) ** 2
        mask = q < 1.0

        phi = torch.zeros_like(q)
        phi_x = torch.zeros_like(q)
        phi_y = torch.zeros_like(q)
        lap_phi = torch.zeros_like(q)

        if mask.any():
            qm = q[mask]
            tm = 1.0 - qm
            phim = torch.exp(-1.0 / tm)

            phi[mask] = phim

            x1m = x1[mask]
            x2m = x2[mask]

            phi_x[mask] = -2.0 * (x1m - cx) / (R**2 * tm**2) * phim
            phi_y[mask] = -2.0 * (x2m - cy) / (R**2 * tm**2) * phim

            lap_phi[mask] = (4.0 * phim / R**2) * (qm * (2.0 * qm - 1.0) / tm**4 - 1.0 / tm**2)

        # oscillation part
        s = torch.sin(K * x1) * torch.sin(K * x2)
        s_x = K * torch.cos(K * x1) * torch.sin(K * x2)
        s_y = K * torch.sin(K * x1) * torch.cos(K * x2)
        lap_s = -2.0 * K**2 * s

        # f = -Δ(u0 + A phi s)
        f_local = -A * (lap_phi * s + 2.0 * (phi_x * s_x + phi_y * s_y) + phi * lap_s)

        return f0 + f_local
    

    def bump(self, x, center=(0.6, 0.4), radius=0.30):
        """
        C^\infty compact-support bump
        x: [N, 2]
        return: [N, 1]
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        cx, cy = center
        r2 = ((x1 - cx) / radius) ** 2 + ((x2 - cy) / radius) ** 2

        out = torch.zeros_like(r2)
        mask = r2 < 1.0
        t = 1.0 - r2[mask]
        out[mask] = torch.exp(-1.0 / t)
        return out
    def exact_solution(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        p1,p2=2,2
        u0 = torch.sin(p1*torch.pi * x1) * torch.sin(p2*torch.pi * x2)

        A = 0.0
        k = 20
        phi = self.bump(x, center=(0.0, 0.0),radius=0.20)

        u_local = A * phi * torch.sin(k * torch.pi * x1) * torch.sin(k * torch.pi * x2)

        return u0 + u_local

    def g(self, x):
        """
        boundary value g(x). u=g at boundary
        x: [N, dim]
        return: [N, 1]
        """
        # example: u=0 at boundary
        return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

    def hard_constraint_func(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        # hf = (1 - x1**2) * (1 - x2**2)
        hf=torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        return hf
        
        
    
    def laplacian_jacrev(self, model_fn, x):
        """
        model_fn: callable, x -> [N,1]
        x: [N, dim]
        return: [N,1]
        """
        def scalar_u(x_single):
            # x_single: [dim]
            y = model_fn(x_single.unsqueeze(0))   # [1,dim] -> [1,1]
            y= y*self.hard_constraint_func(x_single.unsqueeze(0)) # hard constraint
            return y.squeeze()                    # scalar

        def lap_single(x_single):
            H = hessian(scalar_u)(x_single)       # [dim, dim]
            # H=torch.autograd.functional.hessian(scalar_u, x_single, create_graph=True)
            return torch.trace(H)

        lap = vmap(lap_single)(x)                 # [N]
        return lap.unsqueeze(1)                   # [N,1]
        
    def laplacian_autograd(self, model, x):
        """
        适用于普通 loss.backward()
        x: [N, dim]
        return: [N, 1]
        """
        
        x = x.requires_grad_(True)
        u = model(x)   # [N,1]
        u= u * self.hard_constraint_func(x) # hard constraint
        

        grads = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]   # [N, dim]

        lap = 0.0
        for i in range(x.shape[1]):
            grad_i = grads[:, i:i+1]   # [N,1]
            grad2_i = torch.autograd.grad(
                outputs=grad_i,
                inputs=x,
                grad_outputs=torch.ones_like(grad_i),
                create_graph=True,
            )[0][:, i:i+1]   # [N,1]
            lap = lap + grad2_i

        return lap

    def compute_loss(self, model, batch: dict,mode="jacrev"):
        # 3) Total and Weighting
        w_pde = getattr(self.args, "w_pde", 1.0)
        w_bc  = getattr(self.args, "w_bc", 1.0)
        
        # 1) PDE residual
        x_f = batch["X_f"]
        if mode == "backward":
            # print("backward")
            lap_u = self.laplacian_autograd(model, x_f)
            # print(lap_u)
        elif mode == "jacrev":
            # print("jacrev")
            # lap_u=self.gradient_jacrev(model, x_f) #ritz
            lap_u = self.laplacian_jacrev(model, x_f)
            
        f_f = batch.get("f_f", self.f(x_f))

        r_f = (lap_u + f_f)
        loss_pde =  w_pde *0.5*torch.mean(r_f**2)

        # 2) Boundary loss
        x_b = batch["X_b"]
        u_b = model(x_b)
        u_b=u_b*self.hard_constraint_func(x_b)
        g_b = batch.get("g_b", self.g(x_b))
        r_b= u_b-g_b
        loss_bc = 0.5*torch.mean((r_b)**2)*w_bc

  
        total_loss = loss_pde +loss_bc
        r=torch.cat([r_f.flatten(), r_b.flatten()])
        r=r/ math.sqrt(r.numel())
        if mode=="backward":
            r=r.detach()
    # --- 关键修改：返回字典 ---
        loss_dict = {
        "loss":{ 
            "total": total_loss,
            "pde": loss_pde.detach(),
            "bc": loss_bc.detach(),
        },
        "residuals": {
            "all": r
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
        pred = model(grid_xy)
        pred= pred * self.hard_constraint_func(grid_xy) # hard constraint
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

        pred = model(grid_xy)
        pred = pred * self.hard_constraint_func(grid_xy)
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
        
        

    
    @torch.no_grad()
    def plot_gate(self, model, it, save_dir):
        """
        Visualize MoE gate distribution on a 2D grid.
        One subplot per expert.
        """
        # ---- basic config ----
        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # ---- build grid ----
        xs = torch.linspace(low, high, steps=grid_n, device=device, dtype=dtype)
        ys = torch.linspace(low, high, steps=grid_n, device=device, dtype=dtype)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [N,2]

        # ---- compute gate ----
        moe = model.model  # MOE_dense: self.model = self.moe
        with torch.no_grad():
            gates = moe.gating_network(grid_xy)     # [N, E]   

        E = gates.shape[1]

        # ---- plot ----
        fig, axes = plt.subplots(1, E, figsize=(4 * E, 4), squeeze=False)

        for e in range(E):
            gate_e = gates[:, e].reshape(grid_n, grid_n).cpu().numpy()
            ax = axes[0, e]
            im = ax.imshow(
                gate_e,
                origin="lower",
                extent=[low, high, low, high],
                cmap="viridis",
            )
            ax.set_title(f"Gate {e}")
            fig.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"MoE Gate Distribution (iter={it})")
        plt.tight_layout()

        # ---- save ----
        out_dir = save_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gate.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)


