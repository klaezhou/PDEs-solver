# Equation/approximation.py
import os
import math
import torch
import matplotlib.pyplot as plt
from Equation._base import BaseEquation


class FunctionFitEquation(BaseEquation):
    """
    Pure function fitting:
        u(x1, x2) = sin(p1*pi*x1) * sin(p2*pi*x2)

    Domain: [-1, 1]^2
    """

    def __init__(self, args):
        super().__init__(args)
        self.p1 = getattr(args, "p1", 10)
        self.p2 = getattr(args, "p2", 10)

    def target_function(self, x):
        """
        x: [N, 2]
        return: [N, 1]
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        return torch.sin(self.p1 * math.pi * x1) * torch.sin(self.p2 * math.pi * x2)

    def exact_solution(self, x):
        return self.target_function(x)

    def g(self, x):
        """
        Kept only for interface compatibility.
        """
        return self.target_function(x)

    def hard_constraint_func(self, x):
        """
        No hard constraint for pure fitting.
        """
        return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)

    def compute_loss(self, model, batch: dict, mode="backward"):
        """
        Supervised fitting loss:
            loss = 0.5 * mean((u_pred - u_true)^2)

        mode is kept only for compatibility.
        """
        x = batch["X_f"]
        y_true = batch["y_f"]

        y_pred = model(x)
        y_pred = y_pred * self.hard_constraint_func(x)

        r = y_pred - y_true
        loss_fit = 0.5 * torch.mean(r ** 2)

        r_all = r.flatten() / math.sqrt(r.numel())
        if mode == "backward":
            r_all = r_all.detach()

        loss_dict = {
            "loss": {
                "total": loss_fit,
                "fit": loss_fit.detach(),
                "pde": torch.zeros((), device=x.device, dtype=x.dtype),
                "bc": torch.zeros((), device=x.device, dtype=x.dtype),
            },
            "residuals": {
                "all": r_all
            }
        }
        return loss_dict

    def get_data(self, data_loader):
        """
        Sample training points in [-1,1]^2 and generate target values.
        """
        sample_method = getattr(self.args, "sample_method", "grid")

        if sample_method == "random":
            Nf = getattr(self.args, "Nf", 10000)
            X_f = data_loader.sample_interior_box(Nf, dim=2, low=-1.0, high=1.0)

        elif sample_method == "grid":
            nx = getattr(self.args, "nx", 100)
            ny = getattr(self.args, "ny", 100)
            X_f = data_loader.sample_interior_grid_2d(
                nx=nx,
                ny=ny,
                low=-1.0,
                high=1.0,
                exclude_boundary=False
            )
        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")

        y_f = self.target_function(X_f)

        # keep compatibility with old pipeline
        return {
            "X_f": X_f,
            "y_f": y_f,
            "X_b": X_f[:1],   # dummy, avoid some external code crashing
            "g_b": y_f[:1],   # dummy
        }

    @torch.no_grad()
    def plot_error(self, model, it: int, save_dir: str):
        """
        Save pointwise absolute error heatmap and log L2 error.
        """
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

        l2_abs = torch.sqrt(torch.mean(err ** 2)).item()
        denom = torch.sqrt(torch.mean(exact ** 2)).item()
        l2_rel = l2_abs / (denom + eps)

        err_abs_grid = err.abs().reshape(grid_n, grid_n).detach().cpu()

        plt.figure()
        plt.title(f"|u_pred - u_true| (iter={it})\nL2_abs={l2_abs:.3e}, L2_rel={l2_rel:.3e}")
        plt.pcolormesh(X_cpu, Y_cpu, err_abs_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()

        img_path = os.path.join(save_dir, f"fit_error_iter_{it:06d}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        csv_path = os.path.join(save_dir, "fit_error_log.csv")
        need_header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("iter,l2_abs,l2_rel\n")
            f.write(f"{it},{l2_abs:.12e},{l2_rel:.12e}\n")

        print(f"[ErrorPlot] iter={it} | L2_abs={l2_abs:.3e} | L2_rel={l2_rel:.3e} | saved: {img_path}")

        if model_was_training:
            model.train()

    def plot_ground_truth(self, save_dir):
        """
        Plot target function on a 2D grid.
        """
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
        plt.title("Ground Truth")
        plt.pcolormesh(X_cpu, Y_cpu, exact_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()

        out_path = os.path.join(save_dir, "fit_ground_truth.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

    @torch.no_grad()
    def plot_u(self, model, save_dir: str):
        """
        Plot predicted function.
        """
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
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()

        img_path = os.path.join(save_dir, "fit_predict.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        if model_was_training:
            model.train()