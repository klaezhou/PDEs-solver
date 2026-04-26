# Equation/helmholtz.py
import os
import math
import torch
import matplotlib.pyplot as plt
from torch.func import hessian, vmap
from Equation._base import BaseEquation


class HelmholtzEquation(BaseEquation):
    """
    Helmholtz equation:
        Δu + kappa^2 u = f, in Ω=[-1,1]^2
        u = g, on ∂Ω

    exact solution:
        u(x1, x2) = sin(p1*pi*x1) * sin(p2*pi*x2)
    """

    def __init__(self, args):
        super().__init__(args)
        # self.p1 = getattr(args, "p1", 20)
        # self.p2 = getattr(args, "p2", 20)
        self.kappa = getattr(args, "kappa", 1.0)
        ratio = getattr(args, "ratio", 0.8) #(self.kappa * ratio) / math.pi
        self.p1 = 15
        self.p2 = 15

    def exact_solution(self, x):
            # x 形状为 (N, 2)
            x1, x2 = x[:, 0:1], x[:, 1:2]
            return torch.sin(self.p1 * math.pi * x1) * torch.sin(self.p2 * math.pi * x2)

    def f(self, x):
            """
            合理的 RHS 处理方案：
            Δu + kappa^2 u = f
            """
            u = self.exact_solution(x)
            
            # 理论推导：
            # Δu = -( (p1*pi)^2 + (p2*pi)^2 ) * u
            # f = [kappa^2 - ( (p1*pi)^2 + (p2*pi)^2 )] * u
            
            # 使用更为稳定的系数计算
            laplace_coef = (self.p1**2 + self.p2**2) * (math.pi**2)
            coef = self.kappa**2 - laplace_coef
            
            return coef * u

    def g(self, x):
            # 边界条件：直接返回精确解（包含非零边界，更具有普适性）
            return self.exact_solution(x)

    def hard_constraint_func(self, x):
        """
        No hard constraint by default.
        Note: do NOT return Python scalar 1.
        """
        return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)

        # If you want hard zero-Dirichlet constraint, use this instead:
        # x1 = x[:, 0:1]
        # x2 = x[:, 1:2]
        # return (1.0 - x1**2) * (1.0 - x2**2)

    def laplacian_jacrev(self, model_fn, x):
        """
        model_fn: callable, x -> [N,1]
        x: [N, dim]
        return: [N,1]
        """
        def scalar_u(x_single):
            y = model_fn(x_single.unsqueeze(0))   # [1,2] -> [1,1]
            y = y * self.hard_constraint_func(x_single.unsqueeze(0))
            return y.squeeze()

        def lap_single(x_single):
            H = hessian(scalar_u)(x_single)       # [dim, dim]
            return torch.trace(H)

        lap = vmap(lap_single)(x)                 # [N]
        return lap.unsqueeze(1)                   # [N,1]

    def laplacian_autograd(self, model, x):
        """
        For ordinary backward-mode training.
        """
        x = x.requires_grad_(True)
        u = model(x)
        u = u * self.hard_constraint_func(x)

        grads = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]

        lap = 0.0
        for i in range(x.shape[1]):
            grad_i = grads[:, i:i+1]
            grad2_i = torch.autograd.grad(
                outputs=grad_i,
                inputs=x,
                grad_outputs=torch.ones_like(grad_i),
                create_graph=True,
            )[0][:, i:i+1]
            lap = lap + grad2_i

        return lap

    def compute_loss(self, model, batch: dict, mode="jacrev"):
        w_pde = getattr(self.args, "w_pde", 1.0)
        w_bc  = getattr(self.args, "w_bc", 1.0)

        # PDE residual
        x_f = batch["X_f"]
        if mode == "backward":
            lap_u = self.laplacian_autograd(model, x_f)
        elif mode == "jacrev":
            lap_u = self.laplacian_jacrev(model, x_f)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        u_f = model(x_f)
        u_f = u_f * self.hard_constraint_func(x_f)

        f_f = batch.get("f_f", self.f(x_f))

        # Δu + kappa^2 u - f = 0
        r_f = lap_u + self.kappa**2 * u_f - f_f
        loss_pde = 0.5 * torch.mean(r_f**2) * w_pde

        # Boundary residual
        x_b = batch["X_b"]
        u_b = model(x_b)
        u_b = u_b * self.hard_constraint_func(x_b)

        g_b = batch.get("g_b", self.g(x_b))
        r_b = u_b - g_b
        loss_bc = 0.5 * torch.mean(r_b**2) * w_bc

        total_loss = loss_pde + loss_bc

        r = torch.cat([r_f.flatten(), r_b.flatten()])
        r = r / math.sqrt(r.numel())
        if mode == "backward":
            r = r.detach()

        return {
            "loss": {
                "total": total_loss,
                "pde": loss_pde.detach(),
                "bc": loss_bc.detach(),
            },
            "residuals": {
                "all": r
            }
        }

    def get_data(self, data_loader):
        Nf = getattr(self.args, "Nf", 10000)
        Nb = getattr(self.args, "Nb", 2000)

        sample_method = getattr(self.args, "sample_method", "grid")

        if sample_method == "random":
            X_f = data_loader.sample_interior_box(Nf, dim=2, low=-1.0, high=1.0)
            X_b = data_loader.sample_boundary_box_2d(Nb, low=-1.0, high=1.0)
        elif sample_method == "grid":
            nx = getattr(self.args, "nx", 100)
            ny = getattr(self.args, "ny", 100)
            n_per_edge = getattr(self.args, "n_per_edge", 100)
            X_f = data_loader.sample_interior_grid_2d(
                nx=nx, ny=ny, low=-1.0, high=1.0, exclude_boundary=True
            )
            X_b = data_loader.sample_boundary_grid_2d(
                n_per_edge=n_per_edge, low=-1.0, high=1.0, include_corners=True
            )
        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")

        X_f = X_f.requires_grad_(True)
        X_b = X_b.requires_grad_(True)

        f_f = self.f(X_f)
        g_b = self.g(X_b)

        return {
            "X_f": X_f,
            "X_b": X_b,
            "f_f": f_f,
            "g_b": g_b,
        }

    @torch.no_grad()
    def plot_error(self, model, it: int, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = self.args.device
        eps = 1e-12

        cache_key = (grid_n, low, high, str(device))
        if not hasattr(self, "_eval_cache") or self._eval_cache.get("key") != cache_key:
            xs = torch.linspace(low, high, steps=grid_n, device=device)
            ys = torch.linspace(low, high, steps=grid_n, device=device)
            X, Y = torch.meshgrid(xs, ys, indexing="ij")
            grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
            self._eval_cache = {
                "key": cache_key,
                "X_cpu": X.detach().cpu(),
                "Y_cpu": Y.detach().cpu(),
                "grid_xy": grid_xy,
            }

        X_cpu = self._eval_cache["X_cpu"]
        Y_cpu = self._eval_cache["Y_cpu"]
        grid_xy = self._eval_cache["grid_xy"]

        model_was_training = model.training
        model.eval()

        pred = model(grid_xy)
        pred = pred * self.hard_constraint_func(grid_xy)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        exact = self.exact_solution(grid_xy)
        err = pred - exact

        l2_abs = torch.sqrt(torch.mean(err**2)).item()
        denom = torch.sqrt(torch.mean(exact**2)).item()
        l2_rel = l2_abs / (denom + eps)

        err_abs_grid = err.abs().reshape(grid_n, grid_n).detach().cpu()

        plt.figure()
        plt.title(f"|u_pred - u_exact| (iter={it})\nL2_abs={l2_abs:.3e}, L2_rel={l2_rel:.3e}")
        plt.pcolormesh(X_cpu, Y_cpu, err_abs_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        img_path = os.path.join(save_dir, f"helmholtz_error_iter_{it:06d}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        csv_path = os.path.join(save_dir, "helmholtz_error_log.csv")
        need_header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("iter,l2_abs,l2_rel\n")
            f.write(f"{it},{l2_abs:.12e},{l2_rel:.12e}\n")

        print(f"[ErrorPlot] iter={it} | L2_abs={l2_abs:.3e} | L2_rel={l2_rel:.3e} | saved: {img_path}")

        if model_was_training:
            model.train()

    def plot_ground_truth(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = self.args.device

        cache_key = (grid_n, low, high, str(device))
        if not hasattr(self, "_gt_cache") or self._gt_cache.get("key") != cache_key:
            xs = torch.linspace(low, high, steps=grid_n, device=device)
            ys = torch.linspace(low, high, steps=grid_n, device=device)
            X, Y = torch.meshgrid(xs, ys, indexing="ij")
            grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
            self._gt_cache = {
                "key": cache_key,
                "X_cpu": X.detach().cpu(),
                "Y_cpu": Y.detach().cpu(),
                "grid_xy": grid_xy,
            }

        X_cpu = self._gt_cache["X_cpu"]
        Y_cpu = self._gt_cache["Y_cpu"]
        grid_xy = self._gt_cache["grid_xy"]

        exact = self.exact_solution(grid_xy)
        exact_grid = exact.reshape(grid_n, grid_n).detach().cpu()

        plt.figure()
        plt.title("u_exact (ground truth)")
        plt.pcolormesh(X_cpu, Y_cpu, exact_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        out_path = os.path.join(save_dir, "helmholtz_ground_truth.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

    @torch.no_grad()
    def plot_u(self, model, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", -1.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = self.args.device

        cache_key = (grid_n, low, high, str(device))
        if not hasattr(self, "_eval_cache") or self._eval_cache.get("key") != cache_key:
            xs = torch.linspace(low, high, steps=grid_n, device=device)
            ys = torch.linspace(low, high, steps=grid_n, device=device)
            X, Y = torch.meshgrid(xs, ys, indexing="ij")
            grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
            self._eval_cache = {
                "key": cache_key,
                "X_cpu": X.detach().cpu(),
                "Y_cpu": Y.detach().cpu(),
                "grid_xy": grid_xy,
            }

        X_cpu = self._eval_cache["X_cpu"]
        Y_cpu = self._eval_cache["Y_cpu"]
        grid_xy = self._eval_cache["grid_xy"]

        model_was_training = model.training
        model.eval()

        pred = model(grid_xy)
        pred = pred * self.hard_constraint_func(grid_xy)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        u_grid = pred.reshape(grid_n, grid_n).detach().cpu()

        plt.figure()
        plt.title("u_pred")
        plt.pcolormesh(X_cpu, Y_cpu, u_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        img_path = os.path.join(save_dir, "helmholtz_predict.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        if model_was_training:
            model.train()