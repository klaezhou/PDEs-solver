import torch
import numpy as np
def _mask_points_in_any_disk(X: torch.Tensor, centers: torch.Tensor, radius: float):
    """
    判断 X 中每个点是否落在任意一个 center 的圆盘内

    X: [N, d]
    centers: [K, d]
    return:
        mask: [N] bool
        nearest_center_idx: [N]  每个点最近的 center 编号（相对于传入 centers 的局部编号）
    """
    diff = X[:, None, :] - centers[None, :, :]      # [N, K, d]
    dist_sq = (diff ** 2).sum(dim=-1)               # [N, K]

    mask = (dist_sq <= radius ** 2).any(dim=1)      # [N]
    nearest_center_idx = dist_sq.argmin(dim=1)      # [N]

    return mask, nearest_center_idx

def filter_local_data_around_centers(
    data,
    model,
    center_idx,
    radius,
    keep_grad=True,
    return_mask=False,
    min_points_warn=0,
):
    """
    从已有 data 中筛选出落在指定 centers 邻域圆盘内的局部数据

    参数
    ----
    data: dict
        一般包含:
            data["X_f"]: [Nf, d]
            data["X_b"]: [Nb, d]
            data["f_f"]: [Nf, ...]
            data["g_b"]: [Nb, ...]
    model:
        需要有 model.centers, shape = [num_centers, d]
    center_idx: int or list[int]
        要选取的中心编号
    radius: float
        邻域半径
    keep_grad: bool
        是否让返回的 X_f / X_b 保持 requires_grad=True
    return_mask: bool
        是否额外返回 mask 信息
    min_points_warn: int
        若筛出来的点数 <= 该值，就打印警告

    返回
    ----
    local_data: dict
        与原 data 格式一致
    """
    if isinstance(center_idx, int):
        center_idx = [center_idx]

    centers = model.centers[center_idx]   # [K, d]
    
    device = centers.device
    dtype = centers.dtype

    X_f = data["X_f"].detach().to(device=device, dtype=dtype)
    X_b = data["X_b"].detach().to(device=device, dtype=dtype)

    mask_f, owner_f = _mask_points_in_any_disk(X_f, centers, radius)
    mask_b, owner_b = _mask_points_in_any_disk(X_b, centers, radius)

    X_f_local = X_f[mask_f]
    X_b_local = X_b[mask_b]

    if keep_grad:
        X_f_local = X_f_local.clone().requires_grad_(True)
        X_b_local = X_b_local.clone().requires_grad_(True)
    else:
        X_f_local = X_f_local.clone()
        X_b_local = X_b_local.clone()

    local_data = {
        "X_f": X_f_local,
        "X_b": X_b_local,
    }

    # 把对应 target 一起筛出来
    if "f_f" in data and data["f_f"] is not None:
        local_data["f_f"] = data["f_f"].detach().to(device=device, dtype=dtype)[mask_f].clone()

    if "g_b" in data and data["g_b"] is not None:
        local_data["g_b"] = data["g_b"].detach().to(device=device, dtype=dtype)[mask_b].clone()

    # 其他可能想保留的信息

    nf_local = X_f_local.shape[0]
    nb_local = X_b_local.shape[0]

    print(f"[filter_local_data] centers={center_idx}, radius={radius}")
    print(f"[filter_local_data] X_f: {data['X_f'].shape} -> {X_f_local.shape}")
    print(f"[filter_local_data] X_b: {data['X_b'].shape} -> {X_b_local.shape}")

    if nf_local <= min_points_warn:
        print(f"[Warn] local X_f too few: {nf_local}")
    if nb_local <= min_points_warn:
        print(f"[Warn] local X_b too few: {nb_local}")

    if return_mask:
        local_data["mask_f"] = mask_f
        local_data["mask_b"] = mask_b
        local_data["owner_f"] = owner_f[mask_f]   # 每个局部 X_f 属于哪个局部 center
        local_data["owner_b"] = owner_b[mask_b]

    return local_data
