import os
import math
import urllib.request
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from Equation._base import BaseEquation


class Burgers1DEquation(BaseEquation):
    """
    Burgers 1D equation:

        u_t + u u_x = nu u_xx

    Domain:
        x in [-1, 1], t in [0, 1]

    Initial condition:
        u(x,0) = -sin(pi x)

    Boundary condition:
        u(-1,t) = u(1,t) = 0

    Input:
        X[:, 0] = x
        X[:, 1] = t
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.nu = getattr(args, "nu", 0.01 / math.pi)

        self.data_path = getattr(
            args,
            "burgers_data_path",
            "/home/zhy/Zhou/DLPDEs/dlpdes/data/ref/burgers_shock_mu_01_pi.mat"
        )

        self._load_reference_solution()

    # -------------------------------------------------
    # Reference solution
    # -------------------------------------------------
    def _download_reference_solution(self):
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        url = "https://raw.githubusercontent.com/i207M/PINNacle/main/ref/burgers1d.dat"

        if not os.path.exists(self.data_path):
            print(f"[Burgers1D] Downloading reference solution to {self.data_path}")
            urllib.request.urlretrieve(url, self.data_path)

    def _load_reference_solution(self):
        """
        Load burgers1d.dat.

        File format:
            column 0: x
            column 1: u(x,t=0)
            column 2: u(x,t=0.1)
            ...
            column 11: u(x,t=1.0)
        """
        self._download_reference_solution()

        data = np.loadtxt(self.data_path, comments="%")

        self.x_ref = data[:, 0]
        self.t_ref = np.linspace(0.0, 1.0, data.shape[1] - 1)
        self.u_ref = data[:, 1:]

        self.u_interp = RegularGridInterpolator(
            (self.x_ref, self.t_ref),
            self.u_ref,
            bounds_error=False,
            fill_value=None,
        )

        print(
            f"[Burgers1D] Reference loaded: "
            f"x={self.x_ref.shape}, t={self.t_ref.shape}, u={self.u_ref.shape}"
        )

    def exact_solution(self, X):
        """
        Interpolated reference solution.

        X: [N, 2]
        return: [N, 1]
        """
        device = X.device
        dtype = X.dtype

        X_np = X.detach().cpu().numpy()
        u_np = self.u_interp(X_np)

        return torch.tensor(u_np, device=device, dtype=dtype).reshape(-1, 1)

    # -------------------------------------------------
    # IC / BC
    # -------------------------------------------------
    def initial_solution(self, X):
        """
        u(x,0) = -sin(pi x)
        """
        x = X[:, 0:1]
        return -torch.sin(math.pi * x)

    def boundary_solution(self, X):
        """
        u(-1,t) = u(1,t) = 0
        """
        return torch.zeros((X.shape[0], 1), device=X.device, dtype=X.dtype)

    # -------------------------------------------------
    # PDE residual
    # -------------------------------------------------
    def pde_residual_autograd(self, model, X):
        """
        r = u_t + u u_x - nu u_xx
        """
        X = X.requires_grad_(True)

        u = model(X)
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
        u_t = grad_u[:, 1:2]

        grad_u_x = torch.autograd.grad(
            outputs=u_x,
            inputs=X,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_xx = grad_u_x[:, 0:1]

        return u_t + u * u_x - self.nu * u_xx

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------
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
        if u_i.dim() == 1:
            u_i = u_i.unsqueeze(1)

        g_i = batch.get("u_i", self.initial_solution(X_i))
        r_i = u_i - g_i
        loss_ic = 0.5 * torch.mean(r_i ** 2) * w_ic

        # Boundary condition loss
        u_b = model(X_b)
        if u_b.dim() == 1:
            u_b = u_b.unsqueeze(1)

        g_b = batch.get("u_b", self.boundary_solution(X_b))
        r_b = u_b - g_b
        loss_bc = 0.5 * torch.mean(r_b ** 2) * w_bc

        total_loss = loss_pde + loss_ic + loss_bc

        r = torch.cat([
            r_f.flatten(),
            r_i.flatten(),
            r_b.flatten(),
        ])
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

    # -------------------------------------------------
    # Data
    # -------------------------------------------------
    def get_data(self, data_loader=None):
        device = self.args.device
        dtype = torch.float64 if getattr(self.args, "use_double", False) else torch.float32

        Nf = getattr(self.args, "Nf", 10000)
        Ni = getattr(self.args, "Ni", 200)
        Nb = getattr(self.args, "Nb", 200)

        sample_method = getattr(self.args, "sample_method", "random")

        if sample_method == "random":
            # PDE points
            x_f = -1.0 + 2.0 * torch.rand(Nf, 1, device=device, dtype=dtype)
            t_f = torch.rand(Nf, 1, device=device, dtype=dtype)
            X_f = torch.cat([x_f, t_f], dim=1)

            # Initial points: t = 0
            x_i = -1.0 + 2.0 * torch.rand(Ni, 1, device=device, dtype=dtype)
            t_i = torch.zeros_like(x_i)
            X_i = torch.cat([x_i, t_i], dim=1)

            # Boundary points: x = -1 or x = 1
            t_b = torch.rand(Nb, 1, device=device, dtype=dtype)
            side = torch.randint(0, 2, (Nb, 1), device=device)
            x_b = torch.where(
                side == 0,
                -torch.ones_like(t_b),
                torch.ones_like(t_b),
            )
            X_b = torch.cat([x_b, t_b], dim=1)

        elif sample_method == "grid":
            nx = getattr(self.args, "nx", 101)
            nt = getattr(self.args, "nt", 101)

            xs = torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype)
            ts = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)

            X_grid, T_grid = torch.meshgrid(xs, ts, indexing="ij")
            X_all = torch.stack(
                [X_grid.reshape(-1), T_grid.reshape(-1)],
                dim=1,
            )

            mask = (
                (X_all[:, 0] > -1.0)
                & (X_all[:, 0] < 1.0)
                & (X_all[:, 1] > 0.0)
            )
            X_f = X_all[mask]

            x_i = xs.reshape(-1, 1)
            t_i = torch.zeros_like(x_i)
            X_i = torch.cat([x_i, t_i], dim=1)

            t_b = ts.reshape(-1, 1)
            X_left = torch.cat([-torch.ones_like(t_b), t_b], dim=1)
            X_right = torch.cat([torch.ones_like(t_b), t_b], dim=1)
            X_b = torch.cat([X_left, X_right], dim=0)

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

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    @torch.no_grad()
    def plot_error(self, model, it: int, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        nx = getattr(self.args, "eval_nx", 101)
        nt = getattr(self.args, "eval_nt", 101)
        device = self.args.device
        dtype = torch.float64 if getattr(self.args, "use_double", False) else torch.float32
        eps = 1e-12

        xs = torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype)
        ts = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)

        X, T = torch.meshgrid(xs, ts, indexing="ij")
        XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

        model_was_training = model.training
        model.eval()

        pred = model(XT)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        exact = self.exact_solution(XT)
        err = pred - exact

        l2_abs = torch.sqrt(torch.mean(err ** 2)).item()
        denom = torch.sqrt(torch.mean(exact ** 2)).item()
        l2_rel = l2_abs / (denom + eps)

        err_grid = err.abs().reshape(nx, nt).detach().cpu()
        pred_grid = pred.reshape(nx, nt).detach().cpu()
        exact_grid = exact.reshape(nx, nt).detach().cpu()

        X_cpu = X.detach().cpu()
        T_cpu = T.detach().cpu()

        plt.figure()
        plt.title(f"Burgers |error| iter={it}, L2_rel={l2_rel:.3e}")
        plt.pcolormesh(T_cpu, X_cpu, err_grid, shading="auto")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"burgers_error_iter_{it:06d}.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.title("u_pred")
        plt.pcolormesh(T_cpu, X_cpu, pred_grid, shading="auto")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "burgers_predict.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.title("u_exact")
        plt.pcolormesh(T_cpu, X_cpu, exact_grid, shading="auto")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "burgers_exact.png"), dpi=150)
        plt.close()

        csv_path = os.path.join(save_dir, "burgers_error_log.csv")
        need_header = not os.path.exists(csv_path)

        with open(csv_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("iter,l2_abs,l2_rel\n")
            f.write(f"{it},{l2_abs:.12e},{l2_rel:.12e}\n")

        print(
            f"[BurgersError] iter={it} | "
            f"L2_abs={l2_abs:.3e} | L2_rel={l2_rel:.3e}"
        )

        if model_was_training:
            model.train()

    @torch.no_grad()
    def plot_ground_truth(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        nx = getattr(self.args, "eval_nx", 101)
        nt = getattr(self.args, "eval_nt", 101)
        device = self.args.device
        dtype = torch.float64 if getattr(self.args, "use_double", False) else torch.float32

        xs = torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype)
        ts = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)

        X, T = torch.meshgrid(xs, ts, indexing="ij")
        XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

        exact = self.exact_solution(XT)
        exact_grid = exact.reshape(nx, nt).detach().cpu()

        plt.figure()
        plt.title("Burgers reference solution")
        plt.pcolormesh(T.detach().cpu(), X.detach().cpu(), exact_grid, shading="auto")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "burgers_ground_truth.png"), dpi=150)
        plt.close()