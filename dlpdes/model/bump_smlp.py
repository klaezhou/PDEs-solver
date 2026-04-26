import torch
import torch.nn as nn


class LocalBumpTanhNet_Separate(nn.Module):
    """
    Dimension-wise local bump MLP.

    Each input dimension q has its own 1D centers:
        centers_per_dim[q]: [K_q]

    First layer:
        z = W_shared x + b_shared
            + sum_q sum_i psi_{q,i}(x_q) * alpha * (dW_{q,i} * x_q + db_{q,i})

    Hidden layers remain shared MLP.
    """

    def __init__(self, args):
        super().__init__()

        self.depth = getattr(args, "bump_depth", 1)
        self.input_dim = getattr(args, "input_dim", 2)
        self.hidden_dim = getattr(args, "hidden_dim", 60)
        self.eps = getattr(args, "eps", 1e-12)
        self.trainable_centers = getattr(args, "trainable_centers", False)
        self.out_size = getattr(args, "out_size", 1)

        radius = getattr(args, "radius", 0.3)
        self.register_buffer("radius", torch.tensor(float(radius), dtype=torch.float32))

        # ===== per-dimension 1D centers =====
        centers_per_dim = self._build_1d_centers_per_dim(args)
        self.num_centers_per_dim = [c.numel() for c in centers_per_dim]

        self.centers_per_dim = nn.ParameterList()
        self.center_buffers = []

        if self.trainable_centers:
            for c in centers_per_dim:
                self.centers_per_dim.append(nn.Parameter(c))
        else:
            for q, c in enumerate(centers_per_dim):
                name = f"centers_dim_{q}"
                self.register_buffer(name, c)
                self.center_buffers.append(name)

        self.layer_dims = [self.input_dim] + [self.hidden_dim] * self.depth

        # ===== shared layers =====
        self.W_shared = nn.ParameterList()
        self.b_shared = nn.ParameterList()

        for l in range(self.depth):
            in_dim = self.layer_dims[l]
            out_dim = self.layer_dims[l + 1]

            self.W_shared.append(nn.Parameter(torch.empty(out_dim, in_dim)))
            self.b_shared.append(nn.Parameter(torch.empty(out_dim)))

        # ===== dimension-wise local parameters for first layer =====
        # dW_local[q]: [K_q, hidden_dim]
        # db_local[q]: [K_q, hidden_dim]
        self.dW_local = nn.ParameterList()
        self.db_local = nn.ParameterList()

        for q in range(self.input_dim):
            Kq = self.num_centers_per_dim[q]
            self.dW_local.append(nn.Parameter(torch.empty(Kq, self.hidden_dim)))
            self.db_local.append(nn.Parameter(torch.empty(Kq, self.hidden_dim)))

        self.alpha = nn.Parameter(torch.ones(self.hidden_dim), requires_grad=False)

        self.out = nn.Linear(self.hidden_dim, self.out_size, bias=False)

        self.reset_parameters()
        self._report_trainable()

    def _get_centers_q(self, q: int) -> torch.Tensor:
        if self.trainable_centers:
            return self.centers_per_dim[q]
        return getattr(self, f"centers_dim_{q}")

    def reset_parameters(self):
        for l in range(self.depth):
            nn.init.xavier_normal_(self.W_shared[l])
            nn.init.zeros_(self.b_shared[l])

        for q in range(self.input_dim):
            nn.init.zeros_(self.dW_local[q])
            nn.init.zeros_(self.db_local[q])

        nn.init.ones_(self.alpha)
        nn.init.xavier_normal_(self.out.weight)

    def _report_trainable(self, verbose=False):
        total = 0
        group_stats = {}

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            n = p.numel()
            total += n

            key = name.split(".")[0]
            group_stats[key] = group_stats.get(key, 0) + n

            if verbose:
                print(f"{name}: {tuple(p.shape)} -> {n}")

        print("=== Trainable Parameters Summary ===")
        for k, v in group_stats.items():
            print(f"{k:<16} -> {v}")
        print(f"Total trainable params: {total}")
        print(f"Input dim            : {self.input_dim}")
        print(f"Centers per dim      : {self.num_centers_per_dim}")

    def _build_1d_centers_per_dim(self, args):
        """
        Build 1D centers separately for each input dimension.

        Supports:
            args.center_mins  = [min_0, min_1, ...]
            args.center_maxs  = [max_0, max_1, ...]
            args.center_steps = [step_0, step_1, ...]

        Fallback:
            args.center_min / center_max / center_step
        """
        center_mins = getattr(args, "center_mins", None)
        center_maxs = getattr(args, "center_maxs", None)
        center_steps = getattr(args, "center_steps", None)

        if center_mins is None:
            center_min = getattr(args, "center_min", -1.0)
            center_mins = [center_min] * self.input_dim

        if center_maxs is None:
            center_max = getattr(args, "center_max", 1.0)
            center_maxs = [center_max] * self.input_dim

        if center_steps is None:
            center_step = getattr(args, "center_step", 0.1)
            center_steps = [center_step] * self.input_dim

        if len(center_mins) != self.input_dim:
            raise ValueError(
                f"len(center_mins)={len(center_mins)} must equal input_dim={self.input_dim}."
            )

        if len(center_maxs) != self.input_dim:
            raise ValueError(
                f"len(center_maxs)={len(center_maxs)} must equal input_dim={self.input_dim}."
            )

        if len(center_steps) != self.input_dim:
            raise ValueError(
                f"len(center_steps)={len(center_steps)} must equal input_dim={self.input_dim}."
            )

        r = float(self.radius.detach().cpu())
        centers_per_dim = []

        for q in range(self.input_dim):
            cmin = center_mins[q]
            cmax = center_maxs[q]
            cstep = center_steps[q]

            if cstep <= 0:
                raise ValueError(f"center_steps[{q}] must be positive.")

            if cmax < cmin:
                raise ValueError(f"center_maxs[{q}] must be >= center_mins[{q}].")

            axis_q = torch.arange(
                cmin - r,
                cmax + r + 0.5 * cstep,
                cstep
            )

            if axis_q.numel() == 0:
                raise ValueError(f"Generated centers for dim {q} are empty.")

            centers_per_dim.append(axis_q)

        return centers_per_dim

    def _phi_one_dim(self, x_q: torch.Tensor, centers_q: torch.Tensor) -> torch.Tensor:
        """
        x_q:       [B]
        centers_q: [K_q]

        return:
            phi_q: [B, K_q]
        """
        diff = x_q[:, None] - centers_q[None, :]   # [B, K_q]
        dist_sq = diff ** 2

        r_sq = self.radius ** 2
        inside = dist_sq < r_sq

        t = 1.0 - dist_sq / r_sq
        safe_t = torch.where(inside, t, torch.ones_like(t))

        phi_inside = torch.exp(-1.0 / safe_t)
        phi_q = torch.where(inside, phi_inside, torch.zeros_like(t))
        return phi_q

    def _phi_dimwise(self, x: torch.Tensor):
        """
        x: [B, d]

        return:
            phi_list:
                phi_list[q]: [B, K_q]
        """
        phi_list = []

        for q in range(self.input_dim):
            centers_q = self._get_centers_q(q).to(device=x.device, dtype=x.dtype)
            phi_q = self._phi_one_dim(x[:, q], centers_q)
            phi_list.append(phi_q)

        return phi_list

    def bump_weights(self, x: torch.Tensor):
        """
        return:
            psi_list:
                psi_list[q]: [B, K_q]
        """
        phi_list = self._phi_dimwise(x)
        psi_list = []

        for phi_q in phi_list:
            denom = phi_q.sum(dim=1, keepdim=True) + self.eps
            psi_q = phi_q / denom
            psi_list.append(psi_q)

        return psi_list

    def _active_pairs_from_phi_dim(self, phi_q: torch.Tensor):
        """
        phi_q: [B, K_q]

        return:
            b_idx:  [M]
            k_idx:  [M]
            weight: [M]
        """
        denom = phi_q.sum(dim=1, keepdim=True) + self.eps
        psi_q = phi_q / denom

        active = phi_q > 0
        pair_idx = active.nonzero(as_tuple=False)

        if pair_idx.numel() == 0:
            return None, None, None

        b_idx = pair_idx[:, 0]
        k_idx = pair_idx[:, 1]
        weight = psi_q[b_idx, k_idx]
        return b_idx, k_idx, weight

    def _forward_first_layer_dense(self, x: torch.Tensor, psi_list) -> torch.Tensor:
        """
        Dense version.

        x:        [B, d]
        psi_list: list, psi_list[q]: [B, K_q]
        out:      [B, hidden_dim]
        """
        z = x @ self.W_shared[0].t() + self.b_shared[0]  # [B, m]

        local_z = torch.zeros_like(z)

        for q in range(self.input_dim):
            psi_q = psi_list[q]                  # [B, K_q]
            dW_q = self.dW_local[q]              # [K_q, m]
            db_q = self.db_local[q]              # [K_q, m]

            local_w_q = psi_q @ dW_q             # [B, m]
            local_b_q = psi_q @ db_q             # [B, m]

            local_z = local_z + local_w_q * x[:, q:q + 1] + local_b_q

        z = z + self.alpha.view(1, -1) * local_z
        h = torch.tanh(z)
        return h

    def _forward_first_layer_dispatch(self, x: torch.Tensor, phi_list) -> torch.Tensor:
        """
        Dimension-wise sparse dispatch version.

        x:        [B, d]
        phi_list: list, phi_list[q]: [B, K_q]
        out:      [B, hidden_dim]
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        z = x @ self.W_shared[0].t() + self.b_shared[0]  # [B, m]
        local_z = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)

        for q in range(self.input_dim):
            phi_q = phi_list[q]  # [B, K_q]

            b_idx, k_idx, weight = self._active_pairs_from_phi_dim(phi_q)
            if b_idx is None:
                continue

            x_q = x[b_idx, q].unsqueeze(1)          # [M, 1]
            dW_active = self.dW_local[q][k_idx, :]  # [M, m]
            db_active = self.db_local[q][k_idx, :]  # [M, m]

            contrib = dW_active * x_q + db_active   # [M, m]
            contrib = contrib * weight.unsqueeze(1)

            local_z.index_add_(0, b_idx, contrib)

        z = z + self.alpha.view(1, -1) * local_z
        h = torch.tanh(z)
        return h

    def _forward_hidden_layer(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        z = h @ self.W_shared[layer_idx].t() + self.b_shared[layer_idx]
        return torch.tanh(z)

    def forward(self, x: torch.Tensor, use_dispatch: bool = True) -> torch.Tensor:
        """
        x: [B, d]
        y: [B, out_size]
        """
        phi_list = self._phi_dimwise(x)

        if use_dispatch:
            h = self._forward_first_layer_dispatch(x, phi_list)
        else:
            psi_list = [
                phi_q / (phi_q.sum(dim=1, keepdim=True) + self.eps)
                for phi_q in phi_list
            ]
            h = self._forward_first_layer_dense(x, psi_list)

        for l in range(1, self.depth):
            h = self._forward_hidden_layer(h, l)

        y = self.out(h)
        return y

    # -------------------------
    # Freeze / unfreeze helpers
    # -------------------------
    def freeze_all_parameters(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_shared(self, layer_indices=None, with_bias=True):
        if layer_indices is None:
            layer_indices = list(range(self.depth))
        elif isinstance(layer_indices, int):
            layer_indices = [layer_indices]

        layer_set = set(layer_indices)

        for l in range(self.depth):
            flag = l in layer_set
            self.W_shared[l].requires_grad_(flag)
            if with_bias:
                self.b_shared[l].requires_grad_(flag)

    def unfreeze_local(self, dim_indices=None):
        if dim_indices is None:
            dim_indices = list(range(self.input_dim))
        elif isinstance(dim_indices, int):
            dim_indices = [dim_indices]

        dim_set = set(dim_indices)

        for q in range(self.input_dim):
            flag = q in dim_set
            self.dW_local[q].requires_grad_(flag)
            self.db_local[q].requires_grad_(flag)

    def make_local_grad_mask(self, dim_indices=None, center_indices_per_dim=None):
        """
        Build masks for local parameters.

        dim_indices:
            selected input dimensions.
            None means all dimensions.

        center_indices_per_dim:
            None:
                all centers for selected dims.
            dict:
                {q: [center indices for dimension q]}
            list:
                if input_dim=2, e.g. [[0,1,2], [3,4]]
        """
        if dim_indices is None:
            dim_indices = list(range(self.input_dim))
        elif isinstance(dim_indices, int):
            dim_indices = [dim_indices]

        dim_set = set(dim_indices)

        mask_dW_list = []
        mask_db_list = []

        for q in range(self.input_dim):
            mask_dW_q = torch.zeros_like(self.dW_local[q])
            mask_db_q = torch.zeros_like(self.db_local[q])

            if q in dim_set:
                if center_indices_per_dim is None:
                    center_idx_q = list(range(self.num_centers_per_dim[q]))
                elif isinstance(center_indices_per_dim, dict):
                    center_idx_q = center_indices_per_dim.get(
                        q, list(range(self.num_centers_per_dim[q]))
                    )
                else:
                    center_idx_q = center_indices_per_dim[q]

                if isinstance(center_idx_q, int):
                    center_idx_q = [center_idx_q]

                mask_dW_q[center_idx_q, :] = 1.0
                mask_db_q[center_idx_q, :] = 1.0

            mask_dW_list.append(mask_dW_q)
            mask_db_list.append(mask_db_q)

        return mask_dW_list, mask_db_list

    def apply_local_grad_mask_(self, mask_dW_list, mask_db_list):
        for q in range(self.input_dim):
            if self.dW_local[q].grad is not None:
                self.dW_local[q].grad.mul_(mask_dW_list[q])
            if self.db_local[q].grad is not None:
                self.db_local[q].grad.mul_(mask_db_list[q])


def bump_smlp_penultimate_getter(model, x, use_dispatch: bool = True):
    phi_list = model._phi_dimwise(x)

    if use_dispatch:
        h = model._forward_first_layer_dispatch(x, phi_list)
    else:
        psi_list = [
            phi_q / (phi_q.sum(dim=1, keepdim=True) + model.eps)
            for phi_q in phi_list
        ]
        h = model._forward_first_layer_dense(x, psi_list)

    for l in range(1, model.depth):
        h = model._forward_hidden_layer(h, l)

    return h