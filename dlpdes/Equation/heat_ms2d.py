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
import os
import math
import torch
import matplotlib.pyplot as plt
from Equation._base import BaseEquation


class Heat2DMSEquation(BaseEquation):
    """
    2D Heat Multi-Scale equation:

        u_t - a u_xx - b u_yy = 0

    where

        a = 1 / (500*pi)^2
        b = 1 / pi^2

    Domain:
        (x,y,t) in [0,1]^2 x [0,5]

    Initial condition:
        u(x,y,0) = sin(20*pi*x) sin(pi*y)

    Boundary condition:
        u = 0 on spatial boundary.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.a = 1.0 / ((500.0 * math.pi) ** 2)
        self.b = 1.0 / (math.pi ** 2)

        # decay rate
        self.lam = self.a * (20.0 * math.pi) ** 2 + self.b * (math.pi ** 2)

    def exact_solution(self, X):
        """
        X: [N, 3], X[:,0]=x, X[:,1]=y, X[:,2]=t
        return: [N, 1]
        """
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]

        return (
            torch.exp(-self.lam * t)
            * torch.sin(20.0 * math.pi * x)
            * torch.sin(math.pi * y)
        )

    def initial_solution(self, X):
        """
        u(x,y,0) = sin(20*pi*x) sin(pi*y)
        X: [N, 3]
        """
        x = X[:, 0:1]
        y = X[:, 1:2]
        return torch.sin(20.0 * math.pi * x) * torch.sin(math.pi * y)

    def boundary_solution(self, X):
        """
        u = 0 on spatial boundary.
        """
        return torch.zeros((X.shape[0], 1), device=X.device, dtype=X.dtype)

    def hard_constraint_func(self, X):
        """
        Optional hard constraint for spatial boundary.
        If you want hard boundary, use:
            x(1-x)y(1-y)

        Here default returns ones, keeping soft BC.
        """
        return torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)

        # hard version:
        # x = X[:, 0:1]
        # y = X[:, 1:2]
        # return x * (1.0 - x) * y * (1.0 - y)

    def pde_residual_autograd(self, model, X):
        """
        r = u_t - a u_xx - b u_yy
        """
        X = X.requires_grad_(True)

        u = model(X)
        u = u * self.hard_constraint_func(X)

        if u.dim() == 1:
            u = u.unsqueeze(1)

        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]

        grad_u_x = torch.autograd.grad(
            outputs=u_x,
            inputs=X,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_u_y = torch.autograd.grad(
            outputs=u_y,
            inputs=X,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_xx = grad_u_x[:, 0:1]
        u_yy = grad_u_y[:, 1:2]

        r = u_t - self.a * u_xx - self.b * u_yy
        return r

    def compute_loss(self, model, batch: dict, mode="backward"):
        w_pde = getattr(self.args, "w_pde", 1.0)
        w_ic = getattr(self.args, "w_ic", 100.0)
        w_bc = getattr(self.args, "w_bc", 100.0)

        X_f = batch["X_f"]
        X_i = batch["X_i"]
        X_b = batch["X_b"]

        # PDE loss
        r_f = self.pde_residual_autograd(model, X_f)
        loss_pde = 0.5 * torch.mean(r_f ** 2) * w_pde

        # Initial condition loss
        u_i = model(X_i)
        u_i = u_i * self.hard_constraint_func(X_i)
        if u_i.dim() == 1:
            u_i = u_i.unsqueeze(1)

        g_i = batch.get("u_i", self.initial_solution(X_i))
        r_i = u_i - g_i
        loss_ic = 0.5 * torch.mean(r_i ** 2) * w_ic

        # Boundary condition loss
        u_b = model(X_b)
        u_b = u_b * self.hard_constraint_func(X_b)
        if u_b.dim() == 1:
            u_b = u_b.unsqueeze(1)

        g_b = batch.get("u_b", self.boundary_solution(X_b))
        r_b = u_b - g_b
        loss_bc = 0.5 * torch.mean(r_b ** 2) * w_bc

        total_loss = loss_pde + loss_ic + loss_bc

        r = torch.cat([r_f.flatten(), r_i.flatten(), r_b.flatten()])
        r = r / math.sqrt(r.numel())

        if mode == "backward":
            r = r.detach()

        return {
            "loss": {
                "total": total_loss,
                "pde": loss_pde.detach(),
                "ic": loss_ic.detach(),
                "bc": loss_bc.detach(),
            },
            "residuals": {
                "all": r,
            },
        }

    def get_data(self, data_loader=None):
        """
        Generate training data.

        X_f: PDE points in [0,1]^2 x [0,5]
        X_i: initial points at t=0
        X_b: spatial boundary points for t in [0,5]
        """
        device = self.args.device
        dtype = torch.float64 if getattr(self.args, "use_double", False) else torch.float32

        Nf = getattr(self.args, "Nf", 20000)
        Ni = getattr(self.args, "Ni", 2000)
        Nb = getattr(self.args, "Nb", 2000)

        sample_method = getattr(self.args, "sample_method", "random")

        if sample_method == "random":
            # PDE interior
            x_f = torch.rand(Nf, 1, device=device, dtype=dtype)
            y_f = torch.rand(Nf, 1, device=device, dtype=dtype)
            t_f = 5.0 * torch.rand(Nf, 1, device=device, dtype=dtype)
            X_f = torch.cat([x_f, y_f, t_f], dim=1)

            # initial condition: t = 0
            x_i = torch.rand(Ni, 1, device=device, dtype=dtype)
            y_i = torch.rand(Ni, 1, device=device, dtype=dtype)
            t_i = torch.zeros_like(x_i)
            X_i = torch.cat([x_i, y_i, t_i], dim=1)

            # spatial boundary
            t_b = 5.0 * torch.rand(Nb, 1, device=device, dtype=dtype)
            s_b = torch.rand(Nb, 1, device=device, dtype=dtype)

            side = torch.randint(0, 4, (Nb, 1), device=device)

            x_b = torch.zeros_like(s_b)
            y_b = torch.zeros_like(s_b)

            # x=0
            mask = side == 0
            x_b = torch.where(mask, torch.zeros_like(s_b), x_b)
            y_b = torch.where(mask, s_b, y_b)

            # x=1
            mask = side == 1
            x_b = torch.where(mask, torch.ones_like(s_b), x_b)
            y_b = torch.where(mask, s_b, y_b)

            # y=0
            mask = side == 2
            x_b = torch.where(mask, s_b, x_b)
            y_b = torch.where(mask, torch.zeros_like(s_b), y_b)

            # y=1
            mask = side == 3
            x_b = torch.where(mask, s_b, x_b)
            y_b = torch.where(mask, torch.ones_like(s_b), y_b)

            X_b = torch.cat([x_b, y_b, t_b], dim=1)

        elif sample_method == "grid":
            nx = getattr(self.args, "nx", 100)
            ny = getattr(self.args, "ny", 100)
            nt = getattr(self.args, "nt", 50)

            xs = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
            ys = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)
            ts = torch.linspace(0.0, 5.0, nt, device=device, dtype=dtype)

            X, Y, T = torch.meshgrid(xs, ys, ts, indexing="ij")
            X_all = torch.stack(
                [X.reshape(-1), Y.reshape(-1), T.reshape(-1)],
                dim=1,
            )

            mask_f = (
                (X_all[:, 0] > 0.0)
                & (X_all[:, 0] < 1.0)
                & (X_all[:, 1] > 0.0)
                & (X_all[:, 1] < 1.0)
                & (X_all[:, 2] > 0.0)
            )
            X_f = X_all[mask_f]

            # initial
            Xi, Yi = torch.meshgrid(xs, ys, indexing="ij")
            ti = torch.zeros_like(Xi.reshape(-1, 1))
            X_i = torch.cat(
                [Xi.reshape(-1, 1), Yi.reshape(-1, 1), ti],
                dim=1,
            )

            # boundary
            Yb, Tb = torch.meshgrid(ys, ts, indexing="ij")
            X_left = torch.cat(
                [torch.zeros_like(Yb.reshape(-1, 1)), Yb.reshape(-1, 1), Tb.reshape(-1, 1)],
                dim=1,
            )
            X_right = torch.cat(
                [torch.ones_like(Yb.reshape(-1, 1)), Yb.reshape(-1, 1), Tb.reshape(-1, 1)],
                dim=1,
            )

            Xb, Tb2 = torch.meshgrid(xs, ts, indexing="ij")
            Y_bottom = torch.cat(
                [Xb.reshape(-1, 1), torch.zeros_like(Xb.reshape(-1, 1)), Tb2.reshape(-1, 1)],
                dim=1,
            )
            Y_top = torch.cat(
                [Xb.reshape(-1, 1), torch.ones_like(Xb.reshape(-1, 1)), Tb2.reshape(-1, 1)],
                dim=1,
            )

            X_b = torch.cat([X_left, X_right, Y_bottom, Y_top], dim=0)

        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")

        X_f = X_f.requires_grad_(True)
        X_i = X_i.requires_grad_(True)
        X_b = X_b.requires_grad_(True)

        return {
            "X_f": X_f,
            "X_i": X_i,
            "X_b": X_b,
            "u_i": self.initial_solution(X_i),
            "u_b": self.boundary_solution(X_b),
        }

    @torch.no_grad()
    def plot_error(self, model, it: int, save_dir: str):
        """
        Plot error at a fixed time slice.
        Default t_plot = 5.
        """
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        t_plot = getattr(self.args, "t_plot", 5.0)
        device = self.args.device
        dtype = torch.float64 if getattr(self.args, "use_double", False) else torch.float32
        eps = 1e-12

        xs = torch.linspace(0.0, 1.0, grid_n, device=device, dtype=dtype)
        ys = torch.linspace(0.0, 1.0, grid_n, device=device, dtype=dtype)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        T = torch.full_like(X, t_plot)

        grid_xyt = torch.stack(
            [X.reshape(-1), Y.reshape(-1), T.reshape(-1)],
            dim=1,
        )

        model_was_training = model.training
        model.eval()

        pred = model(grid_xyt)
        pred = pred * self.hard_constraint_func(grid_xyt)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        exact = self.exact_solution(grid_xyt)
        err = pred - exact

        l2_abs = torch.sqrt(torch.mean(err ** 2)).item()
        denom = torch.sqrt(torch.mean(exact ** 2)).item()
        l2_rel = l2_abs / (denom + eps)

        err_grid = err.abs().reshape(grid_n, grid_n).detach().cpu()

        plt.figure()
        plt.title(
            f"Heat2D-MS |error| at t={t_plot}, iter={it}\n"
            f"L2_abs={l2_abs:.3e}, L2_rel={l2_rel:.3e}"
        )
        plt.pcolormesh(
            X.detach().cpu(),
            Y.detach().cpu(),
            err_grid,
            shading="auto",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.tight_layout()

        img_path = os.path.join(save_dir, f"heat2d_ms_error_iter_{it:06d}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        csv_path = os.path.join(save_dir, "heat2d_ms_error_log.csv")
        need_header = not os.path.exists(csv_path)

        with open(csv_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("iter,t,l2_abs,l2_rel\n")
            f.write(f"{it},{t_plot},{l2_abs:.12e},{l2_rel:.12e}\n")

        print(
            f"[Heat2D-MS Error] iter={it} | t={t_plot} | "
            f"L2_abs={l2_abs:.3e} | L2_rel={l2_rel:.3e}"
        )

        if model_was_training:
            model.train()
            
    @torch.no_grad()
    def plot_u(self, model, save_dir: str):
        """
        Plot u_pred at fixed time t_plot.
        """
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        t_plot = getattr(self.args, "t_plot", 5.0)
        device = self.args.device
        dtype = torch.float64 if getattr(self.args, "use_double", False) else torch.float32

        cache_key = (grid_n, t_plot, str(device), str(dtype))
        if not hasattr(self, "_eval_cache") or self._eval_cache.get("key") != cache_key:
            xs = torch.linspace(0.0, 1.0, steps=grid_n, device=device, dtype=dtype)
            ys = torch.linspace(0.0, 1.0, steps=grid_n, device=device, dtype=dtype)

            X, Y = torch.meshgrid(xs, ys, indexing="ij")
            T = torch.full_like(X, t_plot)

            grid_xyt = torch.stack(
                [X.reshape(-1), Y.reshape(-1), T.reshape(-1)],
                dim=1,
            )

            self._eval_cache = {
                "key": cache_key,
                "X_cpu": X.detach().cpu(),
                "Y_cpu": Y.detach().cpu(),
                "grid_xyt": grid_xyt,
            }

        X_cpu = self._eval_cache["X_cpu"]
        Y_cpu = self._eval_cache["Y_cpu"]
        grid_xyt = self._eval_cache["grid_xyt"]

        model_was_training = model.training
        model.eval()

        pred = model(grid_xyt)
        pred = pred * self.hard_constraint_func(grid_xyt)

        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        u_grid = pred.reshape(grid_n, grid_n).detach().cpu()

        plt.figure()
        plt.title(f"u_pred at t={t_plot}")
        plt.pcolormesh(X_cpu, Y_cpu, u_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        img_path = os.path.join(save_dir, f"heat2d_ms_predict_t_{t_plot}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        if model_was_training:
            model.train()
            

    @torch.no_grad()
    def plot_ground_truth(self, save_dir: str):
        """
        Plot u_exact at fixed time t_plot.
        """
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        t_plot = getattr(self.args, "t_plot", 5.0)
        device = self.args.device
        dtype = torch.float64 if getattr(self.args, "use_double", False) else torch.float32

        cache_key = (grid_n, t_plot, str(device), str(dtype))
        if not hasattr(self, "_gt_cache") or self._gt_cache.get("key") != cache_key:
            xs = torch.linspace(0.0, 1.0, steps=grid_n, device=device, dtype=dtype)
            ys = torch.linspace(0.0, 1.0, steps=grid_n, device=device, dtype=dtype)

            X, Y = torch.meshgrid(xs, ys, indexing="ij")
            T = torch.full_like(X, t_plot)

            grid_xyt = torch.stack(
                [X.reshape(-1), Y.reshape(-1), T.reshape(-1)],
                dim=1,
            )

            self._gt_cache = {
                "key": cache_key,
                "X_cpu": X.detach().cpu(),
                "Y_cpu": Y.detach().cpu(),
                "grid_xyt": grid_xyt,
            }

        X_cpu = self._gt_cache["X_cpu"]
        Y_cpu = self._gt_cache["Y_cpu"]
        grid_xyt = self._gt_cache["grid_xyt"]

        exact = self.exact_solution(grid_xyt)
        exact_grid = exact.reshape(grid_n, grid_n).detach().cpu()

        plt.figure()
        plt.title(f"u_exact at t={t_plot}")
        plt.pcolormesh(X_cpu, Y_cpu, exact_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        img_path = os.path.join(save_dir, f"heat2d_ms_ground_truth_t_{t_plot}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()