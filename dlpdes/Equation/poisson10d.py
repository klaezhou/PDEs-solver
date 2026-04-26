# Equation/poisson.py
import numpy as np
import torch
import os
from torch.func import functional_call, jacrev, hessian, vmap
from Equation._base import BaseEquation
import math
import matplotlib.pyplot as plt

class POSSION10dEquation(BaseEquation):
    """
    Poisson equation -Δu = f, on domain Ω, with Dirichlet boundary condition u=g at ∂Ω. 
    10-Dimensional case.
    """
    def __init__(self, args):
        super().__init__(args)
        self.dim = 5

    def f(self, x):
            """
            right term f(x) for -Laplace u = f
            x: [N, 10]
            return: [N, 1]
            """
            # 维度 d = 10
            p = 1
            u0 = self.exact_solution(x)
            
            # 对于乘法形式：-Delta u = d * (p * pi)^2 * u
            f0 = self.dim* (p**2 * (math.pi**2)) * u0
            
            return f0

    def exact_solution(self, x):
        """
        x: [N, 10]
        return: [N, 1]
        """
        p = 1
        # 将 torch.sum 改为 torch.prod
        return torch.prod(torch.sin(p * math.pi * x), dim=1, keepdim=True)

    def g(self, x):
        """
        boundary value g(x). u=g at boundary
        x: [N, 10]
        return: [N, 1]
        """
        # 当 p 是整数且定义域在 [-1, 1] 时：
        # 边界上至少有一个 x_i 为 1 或 -1，sin(4 * pi * +/-1) = 0
        # 因此乘法形式在边界上依然恒等于 0
        return self.exact_solution(x)

    def laplacian_jacrev(self, model_fn, x):
        """
        model_fn: callable, x -> [N,1]
        x: [N, dim]
        return: [N,1]
        """
        def scalar_u(x_single):
            # x_single: [dim]
            y = model_fn(x_single.unsqueeze(0))   # [1,dim] -> [1,1]
            return y.squeeze()                    # scalar

        def lap_single(x_single):
            H = hessian(scalar_u)(x_single)       # [dim, dim]
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

    def compute_loss(self, model, batch: dict, mode="jacrev"):
        
        # 1) PDE residual
        x_f = batch["X_f"]
        if mode == "backward":
            lap_u = self.laplacian_autograd(model, x_f)
        elif mode == "jacrev":
            lap_u = self.laplacian_jacrev(model, x_f)
            
        f_f = batch.get("f_f", self.f(x_f))

        r_f = (lap_u + f_f)
        loss_pde = 0.5 * torch.mean(r_f**2)

        # 2) Boundary loss
        x_b = batch["X_b"]
        u_b = model(x_b)
        g_b = batch.get("g_b", self.g(x_b))
        r_b = u_b - g_b
        loss_bc = 0.5 * torch.mean(r_b**2)

        # 3) Total and Weighting
        w_pde = getattr(self.args, "w_pde", 1.0)
        w_bc  = getattr(self.args, "w_bc", 1.0)
        
        total_loss = w_pde * loss_pde + w_bc * loss_bc
        r = torch.cat([r_f.flatten(), r_b.flatten()])
        r = r / math.sqrt(r.numel())
        
        if mode == "backward":
            r = r.detach()
            
        # 返回字典
        loss_dict = {
            "loss": { 
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
        device = getattr(self.args, "device", "cpu")
        
        low = -1.0 
        high = 1.0

        # 2) 10D spatial sampling (Grid sampling is impossible in 10D, forcing random)
        X_f = torch.empty(Nf, self.dim).uniform_(low, high).to(device)
        
        # Boundary sampling: Sample randomly, then push to boundaries for 1 random dimension
        X_b = torch.empty(Nb, self.dim).uniform_(low, high).to(device)
        bnd_dims = torch.randint(0, self.dim, (Nb,))
        # Randomly choose face: -1.0 or 1.0
        bnd_vals = torch.randint(0, 2, (Nb,)) * (high - low) + low 
        X_b[torch.arange(Nb), bnd_dims] = bnd_vals.to(device)

        # 3) precompute f and g values
        X_f = X_f.requires_grad_(True)
        X_b = X_b.requires_grad_(True)
        f_f = self.f(X_f)
        g_b = self.g(X_b)

        # 4) return a unified batch dict
        return {"X_f": X_f, "X_b": X_b, "f_f": f_f, "g_b": g_b}
    
    @torch.no_grad()
    def _build_2d_slice_grid(self, grid_n, low, high, device):
        """
        Helper function to build a 10D grid slice where x1, x2 vary 
        and x3...x10 are fixed to 0.5 (where sin(pi*0.5) = 1) for visualization.
        """
        cache_key = (grid_n, low, high, str(device))
        if not hasattr(self, "_eval_cache") or self._eval_cache.get("key") != cache_key:
            xs = torch.linspace(low, high, steps=grid_n, device=device)
            ys = torch.linspace(low, high, steps=grid_n, device=device)
            X, Y = torch.meshgrid(xs, ys, indexing="ij")
            
            # Base 10D grid fixed at 0.5
            grid_xy = torch.full((grid_n * grid_n, self.dim), 0.5, device=device)
            
            # Allow x1 and x2 to vary across the grid
            grid_xy[:, 0] = X.reshape(-1)
            grid_xy[:, 1] = Y.reshape(-1)
            
            self._eval_cache = {
                "key": cache_key,
                "X_cpu": X.detach().cpu(),
                "Y_cpu": Y.detach().cpu(),
                "grid_xy": grid_xy,
            }
        return self._eval_cache
    
    @torch.no_grad()
    def compute_full_l2_error(self, model, n_mc=None, sampler="sobol"):
        """
        Estimate full-domain dD L2/RMS error by Monte Carlo or Sobol sampling.
        Returns:
            rms_abs, rms_rel, l2_abs, l2_rel
        """
        d = getattr(self.args, "dim", 10)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = getattr(self.args, "device", "cpu")
        eps = 1e-12

        if n_mc is None:
            n_mc = getattr(self.args, "eval_mc_n", 200000)

        model_was_training = model.training
        model.eval()

        if sampler == "sobol":
            eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
            x = eng.draw(n_mc).to(device)  # [0,1]^d
            x = low + (high - low) * x
        else:
            x = low + (high - low) * torch.rand(n_mc, d, device=device)

        pred = model(x)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        exact = self.exact_solution(x)
        if exact.dim() == 1:
            exact = exact.unsqueeze(1)

        err2_mean = torch.mean((pred - exact) ** 2)
        exact2_mean = torch.mean(exact ** 2)

        # domain-normalized RMS
        rms_abs = torch.sqrt(err2_mean).item()
        rms_rel = (torch.sqrt(err2_mean) / (torch.sqrt(exact2_mean) + eps)).item()

        # true mathematical L2 norm
        volume = float((high - low) ** d)
        l2_abs = torch.sqrt(err2_mean * volume).item()
        l2_exact = torch.sqrt(exact2_mean * volume).item()
        l2_rel = l2_abs / (l2_exact + eps)

        if model_was_training:
            model.train()

        return rms_abs, rms_rel, l2_abs, l2_rel

    @torch.no_grad()
    def plot_error(self, model, it: int, save_dir: str):
        """
        Save 2D absolute error heatmap on a fixed slice,
        and log full-domain estimated error.
        """
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = getattr(self.args, "device", "cpu")
        eps = 1e-12

        cache = self._build_2d_slice_grid(grid_n, low, high, device)
        X_cpu, Y_cpu, grid_xy = cache["X_cpu"], cache["Y_cpu"], cache["grid_xy"]

        model_was_training = model.training
        model.eval()

        # ---------- slice prediction ----------
        pred = model(grid_xy)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        exact = self.exact_solution(grid_xy)
        if exact.dim() == 1:
            exact = exact.unsqueeze(1)

        err = pred - exact

        # this is ONLY slice RMS, not full-domain L2
        slice_rms_abs = torch.sqrt(torch.mean(err**2)).item()
        slice_rms_rel = slice_rms_abs / (torch.sqrt(torch.mean(exact**2)).item() + eps)

        err_abs_grid = err.abs().reshape(grid_n, grid_n).detach().cpu()

        # ---------- full-domain error ----------
        full_rms_abs, full_rms_rel, full_l2_abs, full_l2_rel = self.compute_full_l2_error(
            model,
            n_mc=getattr(self.args, "eval_mc_n", 200000),
            sampler=getattr(self.args, "eval_sampler", "sobol"),
        )

        plt.figure()
        plt.title(
            f"|u_pred-u_exact| 10D Slice (iter={it})\n"
            f"slice_RMS={slice_rms_abs:.3e}, full_relL2={full_l2_rel:.3e}"
        )
        plt.pcolormesh(X_cpu, Y_cpu, err_abs_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()

        img_path = os.path.join(save_dir, f"poisson10d_error_iter_{it:06d}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        csv_path = os.path.join(save_dir, "poisson10d_error_log.csv")
        need_header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("iter,slice_rms_abs,slice_rms_rel,full_rms_abs,full_rms_rel,full_l2_abs,full_l2_rel\n")
            f.write(
                f"{it},"
                f"{slice_rms_abs:.12e},{slice_rms_rel:.12e},"
                f"{full_rms_abs:.12e},{full_rms_rel:.12e},"
                f"{full_l2_abs:.12e},{full_l2_rel:.12e}\n"
            )

        print(
            f"[ErrorPlot] iter={it} | "
            f"slice_RMS={slice_rms_abs:.3e} | "
            f"full_RMS={full_rms_abs:.3e} | "
            f"full_relL2={full_l2_rel:.3e} | saved: {img_path}"
        )

        if model_was_training:
            model.train()
            
    def plot_ground_truth(self, save_dir):
        """
        Plot and save the exact solution u_exact on a 2D slice (x3..x10 = 0.5).
        """
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = getattr(self.args, "device", "cpu")

        cache = self._build_2d_slice_grid(grid_n, low, high, device)
        X_cpu, Y_cpu, grid_xy = cache["X_cpu"], cache["Y_cpu"], cache["grid_xy"]

        exact = self.exact_solution(grid_xy)
        exact_grid = exact.reshape(grid_n, grid_n).detach().cpu()

        plt.figure()
        plt.title("u_exact 10D Slice (ground truth)")
        plt.pcolormesh(X_cpu, Y_cpu, exact_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()

        out_path = os.path.join(save_dir, "possion10d_ground_truth.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
    
    @torch.no_grad()
    def plot_u(self, model, save_dir: str):
        """
        Plot the model prediction u_pred on a 2D slice.
        """
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = getattr(self.args, "device", "cpu")

        cache = self._build_2d_slice_grid(grid_n, low, high, device)
        X_cpu, Y_cpu, grid_xy = cache["X_cpu"], cache["Y_cpu"], cache["grid_xy"]

        model_was_training = model.training
        model.eval()

        pred = model(grid_xy)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        u_grid = pred.reshape(grid_n, grid_n).detach().cpu()

        plt.figure()
        plt.title("u_pred 10D Slice")
        plt.pcolormesh(X_cpu, Y_cpu, u_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()

        img_path = os.path.join(save_dir, "poisson10d_predict.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        if model_was_training:
            model.train()
    
    @torch.no_grad()
    def plot_gate(self, model, it, save_dir):
        """
        Visualize MoE gate distribution on a 2D slice (x3..x10 = 0.5).
        """
        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Manual slice build for precision matching dtype
        xs = torch.linspace(low, high, steps=grid_n, device=device, dtype=dtype)
        ys = torch.linspace(low, high, steps=grid_n, device=device, dtype=dtype)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        
        grid_xy = torch.full((grid_n * grid_n, self.dim), 0.5, device=device, dtype=dtype)
        grid_xy[:, 0] = X.reshape(-1)
        grid_xy[:, 1] = Y.reshape(-1)

        moe = model.model
        with torch.no_grad():
            gates = moe.gating_network(grid_xy)     # [N, E]   

        E = gates.shape[1]
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

        fig.suptitle(f"MoE Gate 10D Slice Distribution (iter={it})")
        plt.tight_layout()

        out_dir = save_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "gate.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)