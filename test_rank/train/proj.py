import torch
def min_eigpair_spd(A: torch.Tensor):
    """
    Compute the smallest eigenvalue/eigenvector of an SPD matrix A.

    Args:
        A: (n, n) symmetric positive definite matrix (float32/float64, CPU/GPU)

    Returns:
        (lam_min, v_min) where
        lam_min: scalar tensor
        v_min: (n,) tensor, normalized eigenvector
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square (n, n) matrix.")
    # Optional: enforce symmetry (numerical safety)
    A = 0.5 * (A + A.T)

    evals, evecs = torch.linalg.eigh(A)  # evals ascending
    lam_min = evals[0]
    v_min = evecs[:, 0]  # has normed-1 
    return v_min






def make_jmin_function(model, feature_getter, v):
    # 确保 v 是一个形状为 (m,) 的向量
    v = v.view(-1) 

    def jmin_function(x):
    
        # 1. 准备输入
        x = x.detach().clone().requires_grad_(False)
        x_in = x.unsqueeze(0) if x.ndim == 1 else x
        
        
        # 2. 前向传播获取特征 phi(x)，形状为 [1, m]
        feature_output = feature_getter(model, x_in) 
        
        # 3. 计算标量信号：phi(x) @ v
        # 这就是 JVP 的核心技巧：先做向量内积变成标量，再求导
        # scalar_out 形状为 [1]
        scalar_out = torch.matmul(feature_output, v)

        # 4. 对模型参数求梯度
        # 这里的 inputs 选择了 model.feature 的所有参数
        params = list(model.feature.parameters())
        grads = torch.autograd.grad(
            outputs=scalar_out,
            inputs=params,
            retain_graph=False
        )

        # 5. 将所有参数的梯度展平并拼接成一个长向量 [p]
        # 这样返回的就是一个标准的 1D 向量
        jvp_vector = torch.cat([g.flatten() for g in grads])
        
        return jvp_vector

    return jmin_function


def make_smin_function(model, feature_getter, v):
    # 将 v 展平为长度为 m 的 1D 向量
    v = v.view(-1) 

    def smin_function(x):
        # 1. 准备输入
        x_in = x.unsqueeze(0) if x.ndim == 1 else x
        
        # 2. 前向传播获取特征 phi(x)，形状为 [1, m]
        feature_output = feature_getter(model, x_in) 
        
        # 3. 计算内积：[1, m] @ [m] -> [1]
        # 使用 squeeze() 确保返回的是一个 0 维标量
        smin = torch.matmul(feature_output, v).squeeze()
        
        return smin
    
    return smin_function


def make_grid_2d(nx: int, ny: int, lowx: float, highx: float, lowy: float, highy: float, device=None, dtype=None):
    """
    Create a 2D tensor grid and flatten it.

    Returns:
        x_grid: [N, 2] with N = nx * ny
        X, Y:   [nx, ny] meshgrid (CPU/GPU consistent)
    """
    xs = torch.linspace(lowx, highx, steps=nx, device=device, dtype=dtype)
    ys = torch.linspace(lowy, highy, steps=ny, device=device, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")   # [nx, ny]
    x_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [N,2]

    return x_grid, X, Y

def trapezoid_weights_2d(nx: int, ny: int, lowx: float, highx: float, lowy: float, highy: float, device=None, dtype=None):
    """
    2D trapezoidal rule weights on [lowx, highx] x [lowy, highy].

    Returns:
        w:     [N] weights
        area:  scalar = sum(w) = (highx-lowx) * (highy-lowy)
    """
    dx = (highx - lowx) / (nx - 1)
    dy = (highy - lowy) / (ny - 1)
    wx = torch.ones(nx, device=device, dtype=dtype)
    wy = torch.ones(ny, device=device, dtype=dtype)
    wx[0] *= 0.5
    wx[-1] *= 0.5
    wy[0] *= 0.5
    wy[-1] *= 0.5

    W = torch.outer(wx, wy) * (dx * dy)   # [nx, ny]
    w = W.reshape(-1)                     # [N]
    area = W.sum()
    return w, area

def get_gmin(jmin_f, smin_f, nx, ny, lowx, highx, lowy, highy, device=None, dtype=None):
    # 1. 准备网格和权重
    x_grid, _, _ = make_grid_2d(nx, ny, lowx, highx, lowy, highy, device, dtype)
    w, _ = trapezoid_weights_2d(nx, ny, lowx, highx, lowy, highy, device, dtype)

    # 2. 收集数据 (如果函数不支持批量，此处需循环)
    jvp_list = []
    scalar_list = []
    
    for i in range(x_grid.shape[0]):
        jvp_list.append(jmin_f(x_grid[i]))     # [p]
        scalar_list.append(smin_f(x_grid[i]))   # scalar
    
    JVP_all = torch.stack(jvp_list)      # [N, p]
    scalar_vals = torch.stack(scalar_list) # [N]

    # 3. 执行加权求和: sum_i (JVP_i * s_i * w_i)
    # 我们先计算每个参数分量的积分结果 (形状为 [p])
    # 然后再对参数轴求和得到一个总标量（根据你原代码的要求）
    combined_weights = (scalar_vals * w).unsqueeze(1) # [N, 1]
    integral_per_param = torch.sum(JVP_all * combined_weights, dim=0) # [p]
    
    # 返回所有参数贡献的总和标量
    return integral_per_param

@torch.no_grad()
def mass_model_2d_trapz(
    model,
    feature_getter,
    nx: int, ny: int,
    lowx: float, highx: float,
    lowy: float, highy: float
):
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    # 1) grid
    x_grid, _, _ = make_grid_2d(nx, ny, lowx, highx, lowy, highy, device=device, dtype=dtype)

    # 2) feature getter
    out = feature_getter(model, x_grid)

    if isinstance(out, (tuple, list)) and len(out) == 2:
        Phi, weight = out
    else:
        Phi, weight = out, None

    if Phi.dim() == 1:
        Phi = Phi.unsqueeze(1)

    N, m = Phi.shape
    if N != nx * ny:
        raise RuntimeError(f"Feature shape mismatch: N={N}, nx*ny={nx*ny}")

    # 3) trapz weights
    w, area = trapezoid_weights_2d(nx, ny, lowx, highx, lowy, highy, device=device, dtype=dtype)
    w = w.reshape(-1, 1)
    area = torch.as_tensor(area, device=device, dtype=dtype)

    # 4) build Gram
    if weight is None:
        Phi_w = Phi * w                          # [N,m]
    else:
        # make weight shape compatible with Phi
        if weight.dim() == 1:
            weight = weight.reshape(-1, 1)        # [N,1]
        elif weight.dim() == 2 and weight.shape[1] == 1:
            pass                                 # [N,1]
        # allow [N,m] directly
        if weight.shape[0] != N:
            raise RuntimeError(f"weight N mismatch: weight.shape={weight.shape}, Phi.shape={Phi.shape}")

        # If weight is [N,1], it broadcasts to [N,m]. If weight is [N,m], elementwise works.
        Phi_w = Phi * (weight * w)               # [N,m]

    G = (Phi.T @ Phi_w) / (area + 1e-12)
    return G


def Projection(theta, g):
    """
    Projects vector theta onto the hyperplane orthogonal to g.
    
    Args:
        theta (torch.Tensor): Vector to be projected.
        g (torch.Tensor): Vector defining the hyperplane for projection.
        
    Returns:
        theta_proj (torch.Tensor): The projected vector.
        alpha (torch.Tensor): Scalar that determines the step size for the projection.
    """
    
    # Compute the dot products
    dot_theta_g = torch.matmul(theta, g)  # theta . g
    dot_g_g = torch.matmul(g, g)  # g . g
    epsilon=1e-13
    
    # Compute alpha, ensuring it is non-negative (this could be omitted if you want to allow negative projections)
    alpha = torch.min(-torch.tensor(0.0), dot_theta_g / (dot_g_g+epsilon))  # alpha = max(0, (theta . g) / (g . g))

    # Projection: theta_proj = theta - alpha * g
    theta_proj = theta - alpha * g
    
    return theta_proj, alpha
    
        
def proj_step(theta,model,feature_getter,nx, ny, lowx, highx, lowy, highy):
    device=next(model.parameters()).device
    Mass=mass_model_2d_trapz(model, feature_getter, nx, ny, lowx, highx, lowy, highy)  # [m,m]
    v_min=min_eigpair_spd(Mass)  # [m]
    jmin_f=make_jmin_function(model, feature_getter, v_min)
    smin_f=make_smin_function(model, feature_getter, v_min)
    g_min=get_gmin(jmin_f, smin_f, nx, ny, lowx, highx, lowy, highy,device)  # [p]
    theta_proj, alpha=Projection(theta, g_min)
    return theta_proj, alpha,g_min
    
    