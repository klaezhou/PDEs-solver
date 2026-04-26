import math
import os
import torch
import matplotlib.pyplot as plt
from torch.func import jacrev, hessian, vmap
from Equation._base import BaseEquation


class KovasznayEquation(BaseEquation):
    """
    2D steady incompressible Navier-Stokes (Kovasznay flow) on Omega=[0,1]^2.

    Model output is expected to be [u, v, p], i.e. shape [N, 3].

    PDE:
        u u_x + v u_y = -p_x + nu (u_xx + u_yy)
        u v_x + v v_y = -p_y + nu (v_xx + v_yy)
        u_x + v_y = 0

    Important:
        The usual Kovasznay benchmark uses Dirichlet boundary values taken
        from the exact solution on the boundary, not homogeneous zero BC.
    """

    def _nu(self):
        if hasattr(self.args, "nu") and self.args.nu is not None:
            return float(self.args.nu)
        Re = float(getattr(self.args, "Re", 40.0))
        return 1.0 / Re

    def _Re(self):
        if hasattr(self.args, "Re") and self.args.Re is not None:
            return float(self.args.Re)
        return 1.0 / self._nu()

    def _lambda(self):
        nu = self._nu()
        return 1.0 / (2.0 * nu) - math.sqrt(1.0 / (4.0 * nu * nu) + 4.0 * math.pi * math.pi)

    def f(self, x):
        """
        Optional forcing term for the PDE residual.
        For standard Kovasznay benchmark, forcing is zero.
        return: [N, 3] for (f_u, f_v, f_c)
        """
        return torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)

    def exact_solution(self, x):
        """
        Exact Kovasznay solution.
        x: [N, 2]
        return: [N, 3] -> [u, v, p]
        """
        lam = self._lambda()
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        exp_lx = torch.exp(lam * x1)
        u = 1.0 - exp_lx * torch.cos(2.0 * math.pi * x2)
        v = (lam / (2.0 * math.pi)) * exp_lx * torch.sin(2.0 * math.pi * x2)
        p = 0.5 * (1.0 - torch.exp(2.0 * lam * x1))
        return torch.cat([u, v, p], dim=1)

    def g(self, x):
        """
        Dirichlet boundary data.
        For Kovasznay benchmark, use the exact solution trace on the boundary.
        return: [N, 3]
        """
        return self.exact_solution(x)

    def _first_second_autograd(self, model, x):
        """
        Compute needed first/second derivatives with autograd for backward mode.
        return dict of tensors, each [N,1].
        """
        x = x.requires_grad_(True)
        out = model(x)
        if out.dim() == 1:
            out = out.unsqueeze(1)
        if out.shape[1] != 3:
            raise ValueError(f"KovasznayEquation expects model output [N,3], got {tuple(out.shape)}")

        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]

        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grad_v = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        grad_p = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]
        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

        return {
            "u": u, "v": v, "p": p,
            "u_x": u_x, "u_y": u_y,
            "v_x": v_x, "v_y": v_y,
            "p_x": p_x, "p_y": p_y,
            "u_xx": u_xx, "u_yy": u_yy,
            "v_xx": v_xx, "v_yy": v_yy,
        }

    def _first_second_jacrev(self, model_fn, x):
        """
        Compute needed first/second derivatives with torch.func for jacrev mode.
        model_fn: callable, x -> [N,3]
        x: [N,2]
        """
        def out_single(x_single):
            y = model_fn(x_single.unsqueeze(0))
            if y.dim() == 1:
                return y
            return y.squeeze(0)

        def single_all(x_single):
            y = out_single(x_single)  # [3]
            if y.numel() != 3:
                raise ValueError("KovasznayEquation expects scalar point output with 3 components [u,v,p].")

            J = jacrev(out_single)(x_single)  # [3,2]
            H_u = hessian(lambda xs: out_single(xs)[0])(x_single)  # [2,2]
            H_v = hessian(lambda xs: out_single(xs)[1])(x_single)  # [2,2]

            return (
                y[0].unsqueeze(0),
                y[1].unsqueeze(0),
                y[2].unsqueeze(0),
                J[0, 0].unsqueeze(0),
                J[0, 1].unsqueeze(0),
                J[1, 0].unsqueeze(0),
                J[1, 1].unsqueeze(0),
                J[2, 0].unsqueeze(0),
                J[2, 1].unsqueeze(0),
                H_u[0, 0].unsqueeze(0),
                H_u[1, 1].unsqueeze(0),
                H_v[0, 0].unsqueeze(0),
                H_v[1, 1].unsqueeze(0),
            )

        vals = vmap(single_all)(x)
        keys = [
            "u", "v", "p",
            "u_x", "u_y", "v_x", "v_y", "p_x", "p_y",
            "u_xx", "u_yy", "v_xx", "v_yy",
        ]
        return {k: v for k, v in zip(keys, vals)}

    def compute_loss(self, model, batch: dict, mode="jacrev"):
        w_mom = getattr(self.args, "w_mom", getattr(self.args, "w_pde", 1.0))
        w_cont = getattr(self.args, "w_cont", getattr(self.args, "w_pde", 1.0))
        w_bc = getattr(self.args, "w_bc", 1.0)

        x_f = batch["X_f"]
        if mode == "backward":
            D = self._first_second_autograd(model, x_f)
        elif mode == "jacrev":
            D = self._first_second_jacrev(model, x_f)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        u = D["u"]
        v = D["v"]
        p = D["p"]
        u_x = D["u_x"]
        u_y = D["u_y"]
        v_x = D["v_x"]
        v_y = D["v_y"]
        p_x = D["p_x"]
        p_y = D["p_y"]
        u_xx = D["u_xx"]
        u_yy = D["u_yy"]
        v_xx = D["v_xx"]
        v_yy = D["v_yy"]

        nu = self._nu()
        f_f = batch.get("f_f", self.f(x_f))
        f_u = f_f[:, 0:1]
        f_v = f_f[:, 1:2]
        f_c = f_f[:, 2:3]

        r_mom_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy) - f_u
        r_mom_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy) - f_v
        r_cont = u_x + v_y - f_c

        loss_mom = 0.5 * w_mom * (torch.mean(r_mom_u ** 2) + torch.mean(r_mom_v ** 2))
        loss_cont = 0.5 * w_cont * torch.mean(r_cont ** 2)

        x_b = batch["X_b"]
        pred_b = model(x_b)
        if pred_b.dim() == 1:
            pred_b = pred_b.unsqueeze(1)
        g_b = batch.get("g_b", self.g(x_b))
        r_b = pred_b - g_b
        loss_bc = 0.5 * w_bc * torch.mean(r_b ** 2)

        total_loss = loss_mom + loss_cont + loss_bc

        r_all = torch.cat([
            r_mom_u.reshape(-1),
            r_mom_v.reshape(-1),
            r_cont.reshape(-1),
            r_b.reshape(-1),
        ])
        r_all = r_all / math.sqrt(r_all.numel())
        if mode == "backward":
            r_all = r_all.detach()

        return {
            "loss": {
                "total": total_loss,
                "mom": loss_mom.detach(),
                "cont": loss_cont.detach(),
                "bc": loss_bc.detach(),
            },
            "residuals": {
                "all": r_all,
                "mom_u": r_mom_u.detach() if mode == "backward" else r_mom_u,
                "mom_v": r_mom_v.detach() if mode == "backward" else r_mom_v,
                "cont": r_cont.detach() if mode == "backward" else r_cont,
                "bc": r_b.detach() if mode == "backward" else r_b,
            }
        }

    def get_data(self, data_loader):
        Nf = getattr(self.args, "Nf", 10000)
        Nb = getattr(self.args, "Nb", 2000)
        sample_method = getattr(self.args, "sample_method", "grid")

        low = getattr(self.args, "domain_low", 0.0)
        high = getattr(self.args, "domain_high", 1.0)

        if sample_method == "random":
            X_f = data_loader.sample_interior_box(Nf, dim=2, low=low, high=high)
            X_b = data_loader.sample_boundary_box_2d(Nb, low=low, high=high)
        elif sample_method == "grid":
            nx = getattr(self.args, "nx", 100)
            ny = getattr(self.args, "ny", 100)
            n_per_edge = getattr(self.args, "n_per_edge", 100)
            X_f = data_loader.sample_interior_grid_2d(nx=nx, ny=ny, low=low, high=high, exclude_boundary=True)
            X_b = data_loader.sample_boundary_grid_2d(n_per_edge=n_per_edge, low=low, high=high, include_corners=True)
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
        low = getattr(self.args, "domain_low", 0.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = self.args.device
        eps = 1e-12

        xs = torch.linspace(low, high, steps=grid_n, device=device)
        ys = torch.linspace(low, high, steps=grid_n, device=device)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

        model_was_training = model.training
        model.eval()

        pred = model(grid_xy)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        exact = self.exact_solution(grid_xy)

        # optional pressure gauge alignment for evaluation robustness
        pred_eval = pred.clone()
        pred_eval[:, 2:3] = pred_eval[:, 2:3] - torch.mean(pred_eval[:, 2:3] - exact[:, 2:3])
        err = pred_eval - exact

        l2_rel_u = (torch.sqrt(torch.mean(err[:, 0:1] ** 2)) / (torch.sqrt(torch.mean(exact[:, 0:1] ** 2)) + eps)).item()
        l2_rel_v = (torch.sqrt(torch.mean(err[:, 1:2] ** 2)) / (torch.sqrt(torch.mean(exact[:, 1:2] ** 2)) + eps)).item()
        l2_rel_p = (torch.sqrt(torch.mean(err[:, 2:3] ** 2)) / (torch.sqrt(torch.mean(exact[:, 2:3] ** 2)) + eps)).item()
        l2_rel_all = (torch.sqrt(torch.mean(err ** 2)) / (torch.sqrt(torch.mean(exact ** 2)) + eps)).item()

        err_u = err[:, 0].abs().reshape(grid_n, grid_n).cpu()
        err_v = err[:, 1].abs().reshape(grid_n, grid_n).cpu()
        err_p = err[:, 2].abs().reshape(grid_n, grid_n).cpu()
        X_cpu = X.cpu()
        Y_cpu = Y.cpu()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fields = [(err_u, "|u-u_exact|"), (err_v, "|v-v_exact|"), (err_p, "|p-p_exact| (aligned)")]
        for ax, (field, title) in zip(axes, fields):
            pcm = ax.pcolormesh(X_cpu, Y_cpu, field, shading="auto")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(pcm, ax=ax)

        fig.suptitle(
            f"Kovasznay error (iter={it})\n"
            f"RelL2: u={l2_rel_u:.3e}, v={l2_rel_v:.3e}, p={l2_rel_p:.3e}, all={l2_rel_all:.3e}"
        )
        plt.tight_layout()

        img_path = os.path.join(save_dir, f"kovasznay_error_iter_{it:06d}.png")
        plt.savefig(img_path, dpi=150)
        plt.close(fig)

        csv_path = os.path.join(save_dir, "kovasznay_error_log.csv")
        need_header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("iter,l2_rel_u,l2_rel_v,l2_rel_p,l2_rel_all\n")
            f.write(f"{it},{l2_rel_u:.12e},{l2_rel_v:.12e},{l2_rel_p:.12e},{l2_rel_all:.12e}\n")

        print(
            f"[ErrorPlot] iter={it} | RelL2(u)={l2_rel_u:.3e} | RelL2(v)={l2_rel_v:.3e} | "
            f"RelL2(p)={l2_rel_p:.3e} | RelL2(all)={l2_rel_all:.3e}"
        )

        if model_was_training:
            model.train()

    def plot_ground_truth(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", 0.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = self.args.device

        xs = torch.linspace(low, high, steps=grid_n, device=device)
        ys = torch.linspace(low, high, steps=grid_n, device=device)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

        exact = self.exact_solution(grid_xy)
        u = exact[:, 0].reshape(grid_n, grid_n).cpu()
        v = exact[:, 1].reshape(grid_n, grid_n).cpu()
        p = exact[:, 2].reshape(grid_n, grid_n).cpu()

        X_cpu = X.cpu()
        Y_cpu = Y.cpu()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fields = [(u, "u_exact"), (v, "v_exact"), (p, "p_exact")]
        for ax, (field, title) in zip(axes, fields):
            pcm = ax.pcolormesh(X_cpu, Y_cpu, field, shading="auto")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(pcm, ax=ax)
        plt.tight_layout()

        out_path = os.path.join(save_dir, "kovasznay_ground_truth.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    @torch.no_grad()
    def plot_u(self, model, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        grid_n = getattr(self.args, "eval_grid_n", 200)
        low = getattr(self.args, "domain_low", 0.0)
        high = getattr(self.args, "domain_high", 1.0)
        device = self.args.device

        xs = torch.linspace(low, high, steps=grid_n, device=device)
        ys = torch.linspace(low, high, steps=grid_n, device=device)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

        model_was_training = model.training
        model.eval()

        pred = model(grid_xy)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if pred.shape[1] != 3:
            raise ValueError(f"KovasznayEquation expects model output [N,3], got {tuple(pred.shape)}")

        u_grid = pred[:, 0].reshape(grid_n, grid_n).cpu()

        plt.figure()
        plt.title("u_pred")
        plt.pcolormesh(X.cpu(), Y.cpu(), u_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        img_path = os.path.join(save_dir, "kovasznay_u_predict.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        if model_was_training:
            model.train()
