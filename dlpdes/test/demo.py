import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd.functional import hessian


# ============================================================
# 参数配置
# ============================================================
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

N_interior = 1600
freq = 12
m = 500

adam_lr = 8e-4
adam_epochs = 40000

use_lbfgs = True
lbfgs_lr = 1.0
lbfgs_max_iter = 5000
lbfgs_history_size = 100

save_dir = '/home/zhy/Zhou/DLPDEs/dlpdes/test'
os.makedirs(save_dir, exist_ok=True)

torch.manual_seed(0)


# ============================================================
# 精确解和 RHS
# 一维问题: u'' = f_k(x)
# f_k(x) = sin(k*pi*x)
# u_exact(x) = -sin(k*pi*x)/(k*pi)^2
# ============================================================
def u_exact(x):
    coef = (freq * math.pi) ** 2 #-(1.0 / coef) *
    return  -(1.0 / coef) *torch.sin(freq * math.pi * x[:, 0:1])


def f_rhs(x):
    coef = (freq * math.pi) ** 2
    return torch.sin(freq * math.pi * x[:, 0:1])


# ============================================================
# 数据
# ============================================================
x_in = torch.rand(N_interior, 1, device=device) * 2 - 1
x_in.requires_grad_(True)

x_b = torch.tensor([[-1.0], [1.0]], device=device)
x_b.requires_grad_(True)


# ============================================================
# 单层 tanh 网络
# ============================================================
class SingleLayerTanh(nn.Module):
    def __init__(self, hidden=m):
        super().__init__()
        self.a = nn.Parameter(torch.randn(hidden, 1) * 0.1)
        self.w = nn.Parameter(torch.randn(hidden, 1) * 0.1)
        self.b = nn.Parameter(torch.randn(hidden, 1) * 0.0)

    def forward(self, x):
        z = x @ self.w.T + self.b.T
        return (torch.tanh(z) @ self.a).reshape(-1, 1)


model = SingleLayerTanh(hidden=m).to(device)


# ============================================================
# 通用 PINN loss
# ============================================================
def pinn_loss(model, x, x_boundary, create_graph_for_param=True):
    """
    计算一维 PINN loss:
        loss = mean((u'' - f)^2) + mean(u_boundary^2)

    create_graph_for_param=True:
        用于 Adam / LBFGS / Hessian / gradient norm，需要对参数反传。

    create_graph_for_param=False:
        用于画 loss landscape，只需要对 x 求二阶导，不需要对参数反传。
    """
    u = model(x)

    u_x = torch.autograd.grad(
        u.sum(),
        x,
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x.sum(),
        x,
        create_graph=create_graph_for_param
    )[0]

    loss_in = ((u_xx - f_rhs(x)) ** 2).mean()
    loss_b = (model(x_boundary) ** 2).mean()
    loss = loss_in + loss_b

    return loss, loss_in, loss_b


# ============================================================
# 训练：Adam
# ============================================================
def train_adam(model, x_in, x_b, epochs, lr, print_every=1000):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        loss, loss_in, loss_b = pinn_loss(
            model,
            x_in,
            x_b,
            create_graph_for_param=True
        )

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % print_every == 0:
            print(
                f"[Adam] epoch={epoch}, "
                f"loss={loss.item():.6e}, "
                f"pde={loss_in.item():.6e}, "
                f"bd={loss_b.item():.6e}"
            )

    return losses


# ============================================================
# 训练：LBFGS
# ============================================================
def train_lbfgs(model, x_in, x_b, lr=1.0, max_iter=500, history_size=100, print_every=50):
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe"
    )

    losses = []
    counter = {"n": 0}

    def closure():
        optimizer.zero_grad(set_to_none=True)

        loss, loss_in, loss_b = pinn_loss(
            model,
            x_in,
            x_b,
            create_graph_for_param=True
        )

        loss.backward()

        losses.append(loss.item())

        if counter["n"] % print_every == 0:
            print(
                f"[LBFGS] eval={counter['n']}, "
                f"loss={loss.item():.6e}, "
                f"pde={loss_in.item():.6e}, "
                f"bd={loss_b.item():.6e}"
            )

        counter["n"] += 1
        return loss

    optimizer.step(closure)

    return losses


# ============================================================
# 绘图工具
# ============================================================
def save_loss_curve(loss_list, adam_epochs, filename):
    plt.figure(figsize=(7, 5))
    plt.plot(loss_list)
    plt.axvline(adam_epochs, linestyle="--", label="Adam -> LBFGS")
    plt.yscale("log")
    plt.xlabel("Step / closure evaluation")
    plt.ylabel("MSE Loss")
    plt.title("PINN Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()


def plot_1d(x, y, xlabel, ylabel, title, filename, label=None):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, label=label)
    if label is not None:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()


def plot_predict_vs_exact(x, u_pred, u_true):
    plt.figure(figsize=(7, 5))
    plt.plot(x, u_pred, label="u predict")
    plt.plot(x, u_true, "--", label="u exact")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("u predict vs u exact")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "u_predict_vs_exact.png"), dpi=300)
    plt.show()


def plot_2d_contour(B, A, Z, selected_b, selected_a, min_b, min_a, title, cbar_label, filename):
    plt.figure(figsize=(7, 6))
    cf = plt.contourf(B, A, Z, levels=50)
    plt.colorbar(cf, label=cbar_label)
    plt.scatter([selected_b], [selected_a], marker="x", s=80, label="trained point")
    plt.scatter([min_b], [min_a], marker="o", s=50, label="min on grid")
    plt.xlabel(r"$b_j$")
    plt.ylabel(r"$a_j$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()


def plot_2d_heatmap(B, A, Z, selected_b, selected_a, min_b, min_a, title, cbar_label, filename):
    plt.figure(figsize=(7, 6))
    plt.pcolormesh(B, A, Z, shading="auto")
    plt.colorbar(label=cbar_label)
    plt.scatter([selected_b], [selected_a], marker="x", s=80, label="trained point")
    plt.scatter([min_b], [min_a], marker="o", s=50, label="min on grid")
    plt.xlabel(r"$b_j$")
    plt.ylabel(r"$a_j$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()


# ============================================================
# 参数统计
# ============================================================
def print_param_stats(model):
    with torch.no_grad():
        a_avg_l2 = (model.a ** 2).mean().sqrt().item()
        w_max_abs = torch.abs(model.w).max().item()
        b_avg_l2 = (model.b ** 2).mean().sqrt().item()

    print(f"Average L2 norm a: {a_avg_l2:.6f}")
    print(f"max abs w: {w_max_abs:.6f}")
    print(f"Average L2 norm b: {b_avg_l2:.6f}")


# ============================================================
# Hessian 最小特征值
# ============================================================
def flatten_params(model):
    return torch.cat([
        model.a.flatten(),
        model.w.flatten(),
        model.b.flatten()
    ])


def unflatten_params(params_flat):
    idx_a = m
    idx_w = m

    a = params_flat[:idx_a].reshape(m, 1)
    w = params_flat[idx_a:idx_a + idx_w].reshape(m, 1)
    b = params_flat[idx_a + idx_w:].reshape(m, 1)

    return a, w, b


def manual_forward(x, a, w, b):
    z = x @ w.T + b.T
    return (torch.tanh(z) @ a).reshape(-1, 1)


def manual_loss(a, w, b, x, x_boundary):
    u = manual_forward(x, a, w, b)

    u_x = torch.autograd.grad(
        u.sum(),
        x,
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x.sum(),
        x,
        create_graph=True
    )[0]

    loss_in = ((u_xx - f_rhs(x)) ** 2).mean()
    loss_b = (manual_forward(x_boundary, a, w, b) ** 2).mean()

    return loss_in + loss_b


def compute_min_hessian_eig(model):
    params_flat = flatten_params(model).detach().requires_grad_(True)

    def loss_flat(params):
        a, w, b = unflatten_params(params)
        return manual_loss(a, w, b, x_in, x_b)

    H = hessian(loss_flat, params_flat)
    eigvals = torch.linalg.eigvals(H)
    min_eig = eigvals.real.min().item()

    print("Min Hessian eigenvalue (full):", min_eig)

    return min_eig


# ============================================================
# 最后一步梯度 norm
# ============================================================
def print_grad_norm(model):
    model.zero_grad(set_to_none=True)

    loss, _, _ = pinn_loss(
        model,
        x_in,
        x_b,
        create_graph_for_param=True
    )

    loss.backward()

    grad_a_norm = (model.a.grad ** 2).mean().sqrt().item()
    grad_w_norm = (model.w.grad ** 2).mean().sqrt().item()
    grad_b_norm = (model.b.grad ** 2).mean().sqrt().item()

    print(
        f"Grad avg L2 norm |a|: {grad_a_norm:.6f}, "
        f"|w|: {grad_w_norm:.6f}, "
        f"|b|: {grad_b_norm:.6f}"
    )

    print("frequency:", freq)


# ============================================================
# 最终预测与误差
# ============================================================
def evaluate_and_plot(model, grid_n=1000):
    model.eval()

    with torch.no_grad():
        xs = torch.linspace(-1, 1, grid_n, device=device).reshape(-1, 1)

        u_predict = model(xs)
        u_true = u_exact(xs)

        error = u_predict - u_true
        abs_error = torch.abs(error)

        l2_error = torch.sqrt(torch.mean(error ** 2)).item()
        rel_l2_error = torch.sqrt(
            torch.mean(error ** 2) / (torch.mean(u_true ** 2) + 1e-12)
        ).item()
        max_error = abs_error.max().item()

        X_np = xs.cpu().numpy().reshape(-1)
        U_predict = u_predict.cpu().numpy().reshape(-1)
        U_true = u_true.cpu().numpy().reshape(-1)
        Error = error.cpu().numpy().reshape(-1)
        Abs_error = abs_error.cpu().numpy().reshape(-1)

    print("Final L2 error:", l2_error)
    print("Final relative L2 error:", rel_l2_error)
    print("Final max absolute error:", max_error)

    np.save(os.path.join(save_dir, "x_grid.npy"), X_np)
    np.save(os.path.join(save_dir, "u_predict.npy"), U_predict)
    np.save(os.path.join(save_dir, "u_exact.npy"), U_true)
    np.save(os.path.join(save_dir, "error.npy"), Error)
    np.save(os.path.join(save_dir, "abs_error.npy"), Abs_error)

    plot_predict_vs_exact(X_np, U_predict, U_true)

    plot_1d(
        X_np,
        Abs_error,
        xlabel="x",
        ylabel="absolute error",
        title="absolute error",
        filename="absolute_error.png"
    )

    plot_1d(
        X_np,
        Error,
        xlabel="x",
        ylabel="signed error",
        title="signed error",
        filename="signed_error.png"
    )


# ============================================================
# 二维 loss landscape: L(a_j, b_j)
# ============================================================
def compute_ab_loss_landscape(
    model,
    landscape_n=101,
    a_span=0.1,
    b_span=0.5
):
    """
    选定 |w_j| 最大的神经元，画二维 loss landscape:
        L(a_j, b_j)

    当前保持你代码里的范围:
        a_j in [selected_a - 0.1, selected_a + 0.1]
        b_j in [selected_b - 0.5, selected_b + 0.5]

    如果想改成 +-1，直接把 a_span=1.0, b_span=1.0 即可。
    """

    with torch.no_grad():
        neuron_idx = torch.argmax(torch.abs(model.w[:, 0])).item()

        selected_a = model.a[neuron_idx, 0].item()
        selected_w = model.w[neuron_idx, 0].item()
        selected_b = model.b[neuron_idx, 0].item()

    print("Selected neuron index:", neuron_idx)
    print("Selected neuron a_j:", selected_a)
    print("Selected neuron w_j:", selected_w)
    print("Selected neuron b_j:", selected_b)
    print("Selected |w_j|:", abs(selected_w))

    a_values = selected_a + torch.linspace(-a_span, a_span, landscape_n, device=device)
    b_values = selected_b + torch.linspace(-b_span, b_span, landscape_n, device=device)

    loss_landscape = np.zeros((landscape_n, landscape_n))
    loss_in_landscape = np.zeros_like(loss_landscape)
    loss_b_landscape = np.zeros_like(loss_landscape)

    original_a = model.a.detach().clone()
    original_b = model.b.detach().clone()

    requires_grad_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    model.eval()

    try:
        for ia, aj in enumerate(a_values):
            for ib, bj in enumerate(b_values):
                with torch.no_grad():
                    model.a[neuron_idx, 0].copy_(aj)
                    model.b[neuron_idx, 0].copy_(bj)

                x_eval = x_in.detach().clone().requires_grad_(True)

                loss_total, loss_in, loss_b = pinn_loss(
                    model,
                    x_eval,
                    x_b,
                    create_graph_for_param=False
                )

                loss_landscape[ia, ib] = loss_total.detach().cpu().item()
                loss_in_landscape[ia, ib] = loss_in.detach().cpu().item()
                loss_b_landscape[ia, ib] = loss_b.detach().cpu().item()

    finally:
        with torch.no_grad():
            model.a.copy_(original_a)
            model.b.copy_(original_b)

        for p, flag in zip(model.parameters(), requires_grad_flags):
            p.requires_grad_(flag)

    a_values_np = a_values.detach().cpu().numpy()
    b_values_np = b_values.detach().cpu().numpy()

    np.save(os.path.join(save_dir, "ab_landscape_a_values.npy"), a_values_np)
    np.save(os.path.join(save_dir, "ab_landscape_b_values.npy"), b_values_np)
    np.save(os.path.join(save_dir, "ab_landscape_total_loss.npy"), loss_landscape)
    np.save(os.path.join(save_dir, "ab_landscape_pde_loss.npy"), loss_in_landscape)
    np.save(os.path.join(save_dir, "ab_landscape_boundary_loss.npy"), loss_b_landscape)
    # minimum of pde loss
    min_ia, min_ib = np.unravel_index(np.argmin(loss_in_landscape), loss_in_landscape.shape)

    min_a = a_values_np[min_ia]
    min_b = b_values_np[min_ib]
    min_loss = loss_landscape[min_ia, min_ib]

    origin_ia = int(np.argmin(np.abs(a_values_np - selected_a)))
    origin_ib = int(np.argmin(np.abs(b_values_np - selected_b)))
    origin_loss = loss_landscape[origin_ia, origin_ib]

    print("2D landscape selected neuron index:", neuron_idx)
    print("Original selected a_j:", selected_a)
    print("Original selected b_j:", selected_b)
    print("Landscape min a_j:", min_a)
    print("Landscape min b_j:", min_b)
    print("Landscape min total loss:", min_loss)
    print("Landscape total loss at original (a_j, b_j) approx:", origin_loss)

    B_grid, A_grid = np.meshgrid(b_values_np, a_values_np)

    eps = 1e-30
    log_total_loss = np.log10(loss_landscape + eps)
    log_pde_loss = np.log10(loss_in_landscape + eps)
    log_boundary_loss = np.log10(loss_b_landscape + eps)

    plot_2d_contour(
        B_grid,
        A_grid,
        log_total_loss,
        selected_b,
        selected_a,
        min_b,
        min_a,
        title=r"2D loss landscape: $\log_{10} L(a_j,b_j)$",
        cbar_label=r"$\log_{10}(\mathrm{total\ loss})$",
        filename="ab_loss_landscape_total_contour.png"
    )

    plot_2d_heatmap(
        B_grid,
        A_grid,
        log_total_loss,
        selected_b,
        selected_a,
        min_b,
        min_a,
        title=r"2D heatmap: $\log_{10} L(a_j,b_j)$",
        cbar_label=r"$\log_{10}(\mathrm{total\ loss})$",
        filename="ab_loss_landscape_total_heatmap.png"
    )

    plot_2d_contour(
        B_grid,
        A_grid,
        log_pde_loss,
        selected_b,
        selected_a,
        min_b,
        min_a,
        title=r"2D PDE loss landscape: $\log_{10} L_{\mathrm{PDE}}(a_j,b_j)$",
        cbar_label=r"$\log_{10}(\mathrm{PDE\ loss})$",
        filename="ab_loss_landscape_pde_contour.png"
    )

    plot_2d_contour(
        B_grid,
        A_grid,
        log_boundary_loss,
        selected_b,
        selected_a,
        min_b,
        min_a,
        title=r"2D boundary loss landscape: $\log_{10} L_{\mathrm{bd}}(a_j,b_j)$",
        cbar_label=r"$\log_{10}(\mathrm{boundary\ loss})$",
        filename="ab_loss_landscape_boundary_contour.png"
    )


# ============================================================
# 主流程
# ============================================================
loss_list = []

print("========== Adam training ==========")
adam_loss_list = train_adam(
    model,
    x_in,
    x_b,
    epochs=adam_epochs,
    lr=adam_lr,
    print_every=1000
)
loss_list.extend(adam_loss_list)

if use_lbfgs:
    print("========== LBFGS fine-tuning ==========")
    lbfgs_loss_list = train_lbfgs(
        model,
        x_in,
        x_b,
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        history_size=lbfgs_history_size,
        print_every=50
    )
    loss_list.extend(lbfgs_loss_list)

save_loss_curve(
    loss_list,
    adam_epochs=adam_epochs,
    filename="loss_adam_lbfgs.png"
)

print("========== Parameter statistics ==========")
print_param_stats(model)

print("========== Hessian minimum eigenvalue ==========")
compute_min_hessian_eig(model)

print("========== Gradient norm ==========")
print_grad_norm(model)

print("========== Final prediction and error ==========")
evaluate_and_plot(model, grid_n=1000)

print("========== 2D loss landscape ==========")
compute_ab_loss_landscape(
    model,
    landscape_n=101,
    a_span=10.0,
    b_span=10.0,
)