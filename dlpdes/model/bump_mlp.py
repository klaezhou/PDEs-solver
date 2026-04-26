import math
import torch
import torch.nn as nn


# # The `LocalBumpTanhNet` class implements a single-hidden-layer local MLP with normalized
# # bump-function gating in PyTorch.
class LocalBumpTanhNet(nn.Module):
    """
    Single-hidden-layer local MLP with normalized bump-function gating.

    The model represents

        f(x) = sum_{j=1}^m a_j * tanh( sum_{i=1}^N psi_i(x) * ( (w_j^i)^T x + b_j^i ) ) + c

    where psi_i(x) are normalized compactly supported bump weights:
        psi_i(x) = phi_i(x) / (sum_k phi_k(x) + eps)

    and
        phi_i(x) = exp( -1 / (1 - ||x - xi_i||^2 / r^2) ),   if ||x - xi_i|| < r
                 = 0,                                        otherwise
    """

    def __init__(self, args):
        super().__init__()

        # ===== Read hyperparameters from args =====
        self.depth = getattr(args, "bump_depth", 1)
        self.input_dim = getattr(args, "input_dim", 2)
        self.hidden_dim = getattr(args, "hidden_dim", 60)
        self.eps = getattr(args, "eps", 1e-12)
        self.trainable_centers = getattr(args, "trainable_centers", False)
        self.out_size = getattr(args, "out_size", 1)

        radius = getattr(args, "radius", 0.3)
        self.register_buffer("radius", torch.tensor(float(radius), dtype=torch.float32))
        centers = self._build_grid_centers(args)
        self.num_centers = centers.shape[0]

        assert centers.shape == (self.num_centers, self.input_dim), (
            f"Expected centers shape [{self.num_centers}, {self.input_dim}], "
            f"but got {tuple(centers.shape)}"
        )
        assert radius > 0.0, "radius must be positive"

        # ===== Centers =====
        if self.trainable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

        

        # layer dims: [input_dim, hidden_dim, hidden_dim, ...]
        self.layer_dims = [self.input_dim] + [self.hidden_dim] * self.depth

        # ===== per-layer shared/global parameters =====
        self.W_shared = nn.ParameterList()
        self.b_shared = nn.ParameterList()

        # ===== first-layer local residual parameters =====
        # only layer 0 has local parameters in your current design
        self.dW_local = None   # [N, out_dim, in_dim]
        self.db_local = None   # [N, out_dim]

        # ===== per-layer activation scale alpha =====
        self.alpha = nn.ParameterList()

        for l in range(self.depth):
            in_dim = self.layer_dims[l]
            out_dim = self.layer_dims[l + 1]

            W_bar = nn.Parameter(torch.empty(out_dim, in_dim))
            b_bar = nn.Parameter(torch.empty(out_dim))

            self.W_shared.append(W_bar)
            self.b_shared.append(b_bar)

            if l == 0:
                self.dW_local = nn.Parameter(
                    torch.empty(self.num_centers, out_dim, in_dim)
                )
                self.db_local = nn.Parameter(
                    torch.empty(self.num_centers, out_dim)
                )

                alpha = nn.Parameter(torch.ones(out_dim), requires_grad=False)
                self.alpha.append(alpha)

        self.out = nn.Linear(self.hidden_dim, self.out_size,bias=False)

        self.reset_parameters()
        self._report_trainable()

    def reset_parameters(self):
        for l in range(self.depth):
            nn.init.xavier_normal_(self.W_shared[l])
            nn.init.zeros_(self.b_shared[l])

            if l == 0:
                # local residual init = 0
                nn.init.zeros_(self.dW_local)
                nn.init.zeros_(self.db_local)
                nn.init.ones_(self.alpha[l])*0.1

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
            print(f"{k:<12} -> {v}")
        print(f"Total trainable params: {total}")
        print(f"Num centers  : {self.num_centers}")

    def _build_grid_centers(self, args) -> torch.Tensor:
        """
        Automatically generate centers on a Cartesian grid.
        """
        center_min = getattr(args, "center_min", -1.0)
        center_max = getattr(args, "center_max", 1.0)
        center_step = getattr(args, "center_step", 0.1)

        if center_step <= 0:
            raise ValueError("center_step must be positive.")
        if center_max < center_min:
            raise ValueError("center_max must be >= center_min.")

        axis = torch.arange(
            center_min- 1 * self.radius,
            center_max + 1 * self.radius,
            center_step
        )

        if axis.numel() == 0:
            raise ValueError("Generated axis is empty. Check center_min/center_max/center_step.")

        mesh = torch.meshgrid(*([axis] * self.input_dim), indexing="ij")
        centers = torch.stack([g.reshape(-1) for g in mesh], dim=-1)

        return centers

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, d]
        return phi: [B, N]
        """
        diff = x[:, None, :] - self.centers[None, :, :]   # [B, N, d]
        dist_sq = (diff ** 2).sum(dim=-1)                 # [B, N]

        r_sq = self.radius ** 2
        inside = dist_sq < r_sq

        t = 1.0 - dist_sq / r_sq
        safe_t = torch.where(inside, t, torch.ones_like(t))

        phi_inside = torch.exp(-1.0 / safe_t)
        phi = torch.where(inside, phi_inside, torch.zeros_like(t))
        return phi

    def bump_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return normalized bump weights psi(x): [B, N]
        """
        phi = self._phi(x)
        denom = phi.sum(dim=1, keepdim=True) + self.eps
        psi = phi / denom
        return psi

    def _active_pairs_from_phi(self, phi: torch.Tensor):
        """
        phi: [B, N]
        return:
            b_idx:  [M]
            n_idx:  [M]
            weight: [M]   normalized psi on active pairs
        """
        denom = phi.sum(dim=1, keepdim=True) + self.eps
        psi = phi / denom

        active = phi > 0
        pair_idx = active.nonzero(as_tuple=False)   # [M, 2]

        if pair_idx.numel() == 0:
            return None, None, None

        b_idx = pair_idx[:, 0]
        n_idx = pair_idx[:, 1]
        weight = psi[b_idx, n_idx]
        return b_idx, n_idx, weight

    def _forward_first_layer_dense(self, x: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """
        Dense baseline version.
        x:   [B, d]
        psi: [B, N]
        out: [B, hidden_dim]
        """
        alpha = self.alpha[0]               # [out_dim]
        alpha_w = alpha.view(1, -1, 1)      # [1, out_dim, 1]
        alpha_b = alpha.view(1, -1)         # [1, out_dim]

        # [N, out_dim, in_dim], [N, out_dim]
        W_eff = self.W_shared[0].unsqueeze(0) + alpha_w * self.dW_local
        b_eff = self.b_shared[0].unsqueeze(0) + alpha_b * self.db_local

        local_affine = torch.einsum("bd,nmd->bnm", x, W_eff)   # [B, N, m]
        local_affine = local_affine + b_eff.unsqueeze(0)       # [B, N, m]

        z = torch.einsum("bn,bnm->bm", psi, local_affine)      # [B, m]
        h = torch.tanh(z)
        return h

    def _forward_first_layer_dispatch(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Sparse dispatch version for first layer.
        Only active (b, n) pairs with phi[b, n] > 0 are computed.

        x:   [B, d]
        phi: [B, N]  (unnormalized)
        out: [B, hidden_dim]
        """
        B = x.shape[0]
        out_dim = self.hidden_dim

        b_idx, n_idx, weight = self._active_pairs_from_phi(phi)

        if b_idx is None:
            return torch.zeros(B, out_dim, device=x.device, dtype=x.dtype)

        alpha = self.alpha[0]               # [out_dim]
        alpha_w = alpha.view(1, -1, 1)      # [1, out_dim, 1]
        alpha_b = alpha.view(1, -1)         # [1, out_dim]

        # [N, out_dim, in_dim], [N, out_dim]
        W_eff = self.W_shared[0].unsqueeze(0) + alpha_w * self.dW_local
        b_eff = self.b_shared[0].unsqueeze(0) + alpha_b * self.db_local

        # Gather only active pairs
        x_active = x[b_idx]         # [M, in_dim]
        W_active = W_eff[n_idx]     # [M, out_dim, in_dim]
        b_active = b_eff[n_idx]     # [M, out_dim]

        # [M, out_dim]
        local_affine = torch.einsum("mi,moi->mo", x_active, W_active) + b_active

        # weight by normalized psi
        contrib = local_affine * weight.unsqueeze(1)   # [M, out_dim]

        # scatter-add back to batch axis
        z = torch.zeros(B, out_dim, device=x.device, dtype=x.dtype)
        z.index_add_(0, b_idx, contrib)

        h = torch.tanh(z)
        return h

    def _forward_hidden_layer(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Hidden layer l >= 1:
        Since current design has no per-center local params for l>=1,
        this simplifies to a standard shared layer.

        h:   [B, in_dim]
        out: [B, out_dim]
        """
        z = h @ self.W_shared[layer_idx].t() + self.b_shared[layer_idx]
        h_next = torch.tanh(z)
        return h_next

    def forward(self, x: torch.Tensor, use_dispatch: bool = True) -> torch.Tensor:
        """
        x: [B, d]
        y: [B, out_size]
        """
        phi = self._phi(x)   # [B, N]

        if use_dispatch:
            h = self._forward_first_layer_dispatch(x, phi)
        else:
            psi = phi / (phi.sum(dim=1, keepdim=True) + self.eps)
            h = self._forward_first_layer_dense(x, psi)

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

    # def unfreeze_local_centers(self, center_indices):
    #     """
    #     Note:
    #     In this tensor-parameter version, dW_local/db_local are single Parameters.
    #     PyTorch cannot set requires_grad for only slices.

    #     So here we unfreeze the whole tensor, and you should use gradient masking
    #     in training if you want only selected centers to update.
    #     """
    #     self.dW_local.requires_grad_(True)
    #     self.db_local.requires_grad_(True)

    def unfreeze_shared(self, layer_indices=None, with_bias=True):
        if layer_indices is None:
            layer_indices = list(range(self.depth))
        elif isinstance(layer_indices, int):
            layer_indices = [layer_indices]

        layer_set = set(layer_indices)

        for l in range(self.depth):
            flag = (l in layer_set)
            self.W_shared[l].requires_grad_(flag)
            if with_bias:
                self.b_shared[l].requires_grad_(flag)

    # -------------------------
    # Gradient mask helper
    # -------------------------
    # def make_local_grad_mask(self, center_indices):
    #     """
    #     Build masks for dW_local and db_local so you can manually zero out grads
    #     of unselected centers after backward().

    #     return:
    #         mask_dW: [N, out_dim, in_dim]
    #         mask_db: [N, out_dim]
    #     """
    #     if isinstance(center_indices, int):
    #         center_indices = [center_indices]

    #     device = self.dW_local.device
    #     dtype = self.dW_local.dtype

    #     mask_dW = torch.zeros_like(self.dW_local, device=device, dtype=dtype)
    #     mask_db = torch.zeros_like(self.db_local, device=device, dtype=dtype)

    #     mask_dW[center_indices] = 1.0
    #     mask_db[center_indices] = 1.0
    #     return mask_dW, mask_db

    # def apply_local_grad_mask_(self, mask_dW, mask_db):
    #     """
    #     Call this after loss.backward() and before optimizer.step()
    #     """
    #     if self.dW_local.grad is not None:
    #         self.dW_local.grad.mul_(mask_dW)
    #     if self.db_local.grad is not None:
    #         self.db_local.grad.mul_(mask_db)
    

def bump_mlp_penultimate_getter(model, x, use_dispatch: bool = True):
    """
    Get the penultimate layer activations from the bump model.

    Parameters:
    - model: LocalBumpTanhNet
    - x:     [B, input_dim]

    Returns:
    - h:     [B, hidden_dim]
    """
    phi = model._phi(x)

    if use_dispatch:
        h = model._forward_first_layer_dispatch(x, phi)
    else:
        psi = phi / (phi.sum(dim=1, keepdim=True) + model.eps)
        h = model._forward_first_layer_dense(x, psi)

    for l in range(1, model.depth):
        h = model._forward_hidden_layer(h, l)

    return h



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# class LocalBumpTanhNet(nn.Module):
#     """
#     Multi-layer local MLP with normalized bump-function gating.

#     For each layer l and center i:
#         W_eff^(l,i) = W_shared^(l) + alpha^(l) * dW_local^(l,i)
#         b_eff^(l,i) = b_shared^(l) + alpha^(l) * db_local^(l,i)

#     and the layer output is
#         z^(l)(x) = sum_i psi_i(x) * [ W_eff^(l,i) h^(l-1)(x) + b_eff^(l,i) ]
#         h^(l)(x) = tanh(z^(l)(x))
#     """

#     def __init__(self, args):
#         super().__init__()

#         # ===== Read hyperparameters from args =====
#         self.depth = getattr(args, "bump_depth", 1)
#         self.input_dim = getattr(args, "input_dim", 2)
#         self.hidden_dim = getattr(args, "hidden_dim", 60)
#         self.eps = getattr(args, "eps", 1e-12)
#         self.trainable_centers = getattr(args, "trainable_centers", False)
#         self.out_size = getattr(args, "out_size", 1)

#         radius = getattr(args, "radius", 0.3)

#         centers = self._build_grid_centers(args)
#         self.num_centers = centers.shape[0]

#         assert centers.shape == (self.num_centers, self.input_dim), (
#             f"Expected centers shape [{self.num_centers}, {self.input_dim}], "
#             f"but got {tuple(centers.shape)}"
#         )
#         assert radius > 0.0, "radius must be positive"

#         # ===== Centers =====
#         if self.trainable_centers:
#             self.centers = nn.Parameter(centers)
#         else:
#             self.register_buffer("centers", centers)

#         self.register_buffer("radius", torch.tensor(float(radius), dtype=torch.float32))

#         # layer dims: [input_dim, hidden_dim, hidden_dim, ...]
#         self.layer_dims = [self.input_dim] + [self.hidden_dim] * self.depth

#         # ===== per-layer shared/global parameters =====
#         self.W_shared = nn.ParameterList()
#         self.b_shared = nn.ParameterList()

#         # ===== per-layer local residual parameters =====
#         # each layer l has:
#         #   dW_local[l]: [N, out_dim, in_dim]
#         #   db_local[l]: [N, out_dim]
#         self.dW_local = nn.ParameterList()
#         self.db_local = nn.ParameterList()

#         # ===== per-layer activation scale alpha =====
#         self.alpha = nn.ParameterList()

#         for l in range(self.depth):
#             in_dim = self.layer_dims[l]
#             out_dim = self.layer_dims[l + 1]

#             W_bar = nn.Parameter(torch.empty(out_dim, in_dim))
#             b_bar = nn.Parameter(torch.empty(out_dim))
#             self.W_shared.append(W_bar)
#             self.b_shared.append(b_bar)

#             dW = nn.Parameter(torch.empty(self.num_centers, out_dim, in_dim))
#             db = nn.Parameter(torch.empty(self.num_centers, out_dim))
#             self.dW_local.append(dW)
#             self.db_local.append(db)

#             # keep same style as your previous code: alpha not trainable by default
#             alpha = nn.Parameter(torch.ones(out_dim), requires_grad=False)
#             self.alpha.append(alpha)

#         self.out = nn.Linear(self.hidden_dim, self.out_size, bias=False)

#         self.reset_parameters()
#         self._report_trainable()

#     def reset_parameters(self):
#         for l in range(self.depth):
#             nn.init.xavier_normal_(self.W_shared[l])
#             nn.init.zeros_(self.b_shared[l])

#             # local residual init = 0
#             nn.init.zeros_(self.dW_local[l])
#             nn.init.zeros_(self.db_local[l])
#             nn.init.ones_(self.alpha[l])

#         nn.init.xavier_normal_(self.out.weight)

#     def _report_trainable(self, verbose=False):
#         total = 0
#         group_stats = {}

#         for name, p in self.named_parameters():
#             if not p.requires_grad:
#                 continue

#             n = p.numel()
#             total += n

#             key = name.split(".")[0]
#             group_stats[key] = group_stats.get(key, 0) + n

#             if verbose:
#                 print(f"{name}: {tuple(p.shape)} -> {n}")

#         print("=== Trainable Parameters Summary ===")
#         for k, v in group_stats.items():
#             print(f"{k:<12} -> {v}")
#         print(f"Total trainable params: {total}")
#         print(f"Num centers  : {self.num_centers}")

#     def _build_grid_centers(self, args) -> torch.Tensor:
#         center_min = getattr(args, "center_min", -1.0)
#         center_max = getattr(args, "center_max", 1.0)
#         center_step = getattr(args, "center_step", 0.1)

#         if center_step <= 0:
#             raise ValueError("center_step must be positive.")
#         if center_max < center_min:
#             raise ValueError("center_max must be >= center_min.")

#         axis = torch.arange(
#             center_min,
#             center_max + 0.5 * center_step,
#             center_step
#         )

#         if axis.numel() == 0:
#             raise ValueError("Generated axis is empty. Check center_min/center_max/center_step.")

#         mesh = torch.meshgrid(*([axis] * self.input_dim), indexing="ij")
#         centers = torch.stack([g.reshape(-1) for g in mesh], dim=-1)
#         return centers

#     def _phi(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, d]
#         return phi: [B, N]
#         """
#         diff = x[:, None, :] - self.centers[None, :, :]   # [B, N, d]
#         dist_sq = (diff ** 2).sum(dim=-1)                 # [B, N]

#         r_sq = self.radius ** 2
#         inside = dist_sq < r_sq

#         t = 1.0 - dist_sq / r_sq
#         safe_t = torch.where(inside, t, torch.ones_like(t))

#         phi_inside = torch.exp(-1.0 / safe_t)
#         phi = torch.where(inside, phi_inside, torch.zeros_like(t))
#         return phi

#     def bump_weights(self, x: torch.Tensor) -> torch.Tensor:
#         phi = self._phi(x)
#         denom = phi.sum(dim=1, keepdim=True) + self.eps
#         psi = phi / denom
#         return psi

#     def _active_pairs_from_phi(self, phi: torch.Tensor):
#         """
#         phi: [B, N]
#         return:
#             b_idx:  [M]
#             n_idx:  [M]
#             weight: [M]   normalized psi on active pairs
#         """
#         denom = phi.sum(dim=1, keepdim=True) + self.eps
#         psi = phi / denom

#         active = phi > 0
#         pair_idx = active.nonzero(as_tuple=False)   # [M, 2]

#         if pair_idx.numel() == 0:
#             return None, None, None

#         b_idx = pair_idx[:, 0]
#         n_idx = pair_idx[:, 1]
#         weight = psi[b_idx, n_idx]
#         return b_idx, n_idx, weight

#     def _layer_params(self, layer_idx: int):
#         """
#         Returns:
#             W_eff: [N, out_dim, in_dim]
#             b_eff: [N, out_dim]
#         """
#         alpha = self.alpha[layer_idx]               # [out_dim]
#         alpha_w = alpha.view(1, -1, 1)              # [1, out_dim, 1]
#         alpha_b = alpha.view(1, -1)                 # [1, out_dim]

#         W_eff = self.W_shared[layer_idx].unsqueeze(0) + alpha_w * self.dW_local[layer_idx]
#         b_eff = self.b_shared[layer_idx].unsqueeze(0) + alpha_b * self.db_local[layer_idx]
#         return W_eff, b_eff

#     def _forward_layer_dense(self, h_in: torch.Tensor, psi: torch.Tensor, layer_idx: int) -> torch.Tensor:
#         """
#         Dense version for any layer.

#         h_in: [B, in_dim]
#         psi:  [B, N]
#         out:  [B, out_dim]
#         """
#         W_eff, b_eff = self._layer_params(layer_idx)                 # [N, out, in], [N, out]

#         local_affine = torch.einsum("bi,noi->bno", h_in, W_eff)      # [B, N, out]
#         local_affine = local_affine + b_eff.unsqueeze(0)             # [B, N, out]

#         z = torch.einsum("bn,bno->bo", psi, local_affine)            # [B, out]
#         h_out = torch.tanh(z)
#         return h_out

#     def _forward_layer_dispatch(
#         self,
#         h_in: torch.Tensor,
#         phi: torch.Tensor,
#         layer_idx: int,
#         cached_active=None,
#     ) -> torch.Tensor:
#         """
#         Sparse dispatch version for any layer.
#         Only active (b, n) pairs with phi[b, n] > 0 are computed.

#         h_in: [B, in_dim]
#         phi:  [B, N]
#         out:  [B, out_dim]
#         """
#         B = h_in.shape[0]
#         out_dim = self.layer_dims[layer_idx + 1]

#         if cached_active is None:
#             b_idx, n_idx, weight = self._active_pairs_from_phi(phi)
#         else:
#             b_idx, n_idx, weight = cached_active

#         if b_idx is None:
#             return torch.zeros(B, out_dim, device=h_in.device, dtype=h_in.dtype)

#         W_eff, b_eff = self._layer_params(layer_idx)   # [N, out, in], [N, out]

#         h_active = h_in[b_idx]         # [M, in_dim]
#         W_active = W_eff[n_idx]        # [M, out_dim, in_dim]
#         b_active = b_eff[n_idx]        # [M, out_dim]

#         local_affine = torch.einsum("mi,moi->mo", h_active, W_active) + b_active   # [M, out_dim]
#         contrib = local_affine * weight.unsqueeze(1)                                # [M, out_dim]

#         z = torch.zeros(B, out_dim, device=h_in.device, dtype=h_in.dtype)
#         z.index_add_(0, b_idx, contrib)

#         h_out = torch.tanh(z)
#         return h_out

#     def forward(self, x: torch.Tensor, use_dispatch: bool = True) -> torch.Tensor:
#         """
#         x: [B, d]
#         y: [B, out_size]
#         """
#         phi = self._phi(x)   # [B, N]

#         if use_dispatch:
#             cached_active = self._active_pairs_from_phi(phi)
#             h = x
#             for l in range(self.depth):
#                 h = self._forward_layer_dispatch(h, phi, l, cached_active=cached_active)
#         else:
#             psi = phi / (phi.sum(dim=1, keepdim=True) + self.eps)
#             h = x
#             for l in range(self.depth):
#                 h = self._forward_layer_dense(h, psi, l)

#         y = self.out(h)
#         return y

#     # -------------------------
#     # Freeze / unfreeze helpers
#     # -------------------------
#     def freeze_all_parameters(self):
#         for p in self.parameters():
#             p.requires_grad_(False)

#     def unfreeze_shared(self, layer_indices=None, with_bias=True):
#         if layer_indices is None:
#             layer_indices = list(range(self.depth))
#         elif isinstance(layer_indices, int):
#             layer_indices = [layer_indices]

#         layer_set = set(layer_indices)

#         for l in range(self.depth):
#             flag = (l in layer_set)
#             self.W_shared[l].requires_grad_(flag)
#             if with_bias:
#                 self.b_shared[l].requires_grad_(flag)

#     def unfreeze_local_layers(self, layer_indices=None):
#         """
#         Unfreeze local tensors of selected layers.
#         Note: entire tensor of that layer is unfrozen, not partial centers.
#         """
#         if layer_indices is None:
#             layer_indices = list(range(self.depth))
#         elif isinstance(layer_indices, int):
#             layer_indices = [layer_indices]

#         layer_set = set(layer_indices)

#         for l in range(self.depth):
#             flag = (l in layer_set)
#             self.dW_local[l].requires_grad_(flag)
#             self.db_local[l].requires_grad_(flag)

#     def make_local_grad_mask(self, center_indices, layer_indices=None):
#         """
#         Build grad masks for selected centers on selected layers.

#         Returns:
#             mask_dW_list: list of masks, each [N, out_dim, in_dim]
#             mask_db_list: list of masks, each [N, out_dim]
#         """
#         if isinstance(center_indices, int):
#             center_indices = [center_indices]

#         if layer_indices is None:
#             layer_indices = list(range(self.depth))
#         elif isinstance(layer_indices, int):
#             layer_indices = [layer_indices]

#         layer_set = set(layer_indices)

#         mask_dW_list = []
#         mask_db_list = []

#         for l in range(self.depth):
#             mask_dW = torch.zeros_like(self.dW_local[l])
#             mask_db = torch.zeros_like(self.db_local[l])

#             if l in layer_set:
#                 mask_dW[center_indices] = 1.0
#                 mask_db[center_indices] = 1.0

#             mask_dW_list.append(mask_dW)
#             mask_db_list.append(mask_db)

#         return mask_dW_list, mask_db_list

#     def apply_local_grad_mask_(self, mask_dW_list, mask_db_list):
#         for l in range(self.depth):
#             if self.dW_local[l].grad is not None:
#                 self.dW_local[l].grad.mul_(mask_dW_list[l])
#             if self.db_local[l].grad is not None:
#                 self.db_local[l].grad.mul_(mask_db_list[l])


# def bump_mlp_penultimate_getter(model, x, use_dispatch: bool = True):
#     """
#     Returns the last hidden feature before output layer.
#     """
#     phi = model._phi(x)

#     if use_dispatch:
#         cached_active = model._active_pairs_from_phi(phi)
#         h = x
#         for l in range(model.depth):
#             h = model._forward_layer_dispatch(h, phi, l, cached_active=cached_active)
#     else:
#         psi = phi / (phi.sum(dim=1, keepdim=True) + model.eps)
#         h = x
#         for l in range(model.depth):
#             h = model._forward_layer_dense(h, psi, l)

#     return h