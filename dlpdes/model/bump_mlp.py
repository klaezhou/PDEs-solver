import math
import torch
import torch.nn as nn


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

    def __init__(
        self,
        args
    ):
        super().__init__()

        # ===== Read hyperparameters from args =====
        self.depth = getattr(args, "bump_depth", 1)
        self.input_dim = getattr(args, "input_dim", 2)
        # self.num_centers = getattr(args, "num_centers", 4)
        self.hidden_dim = getattr(args, "hidden_dim", 30)
        self.eps = getattr(args, "eps", 1e-12)
        self.trainable_centers = getattr(args, "trainable_centers", False)

        radius = getattr(args, "radius", 0.3) #0.05
        # centers = getattr(args, "centers", None) # torch.tensor


        # centers = centers.clone().float()
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

        # Keep radius as a scalar buffer.
        self.register_buffer("radius", torch.tensor(float(radius), dtype=torch.float32))

        # Trainable local parameters:
        # W_local[i, j, :] = w_j^i
        # b_local[i, j]    = b_j^i
        self.layer_dims = [self.input_dim] + [self.hidden_dim] * self.depth

        # ===== per-layer shared/global parameters =====
        self.W_shared = nn.ParameterList()
        self.b_shared = nn.ParameterList()

        # ===== per-layer local residual parameters =====
        # self.dW_local = nn.ParameterList()
        # self.db_local = nn.ParameterList()
        self.dW_local = nn.ModuleList()
        self.db_local = nn.ModuleList()
        # ===== per-layer activation scale alpha =====
        self.alpha = nn.ParameterList()
        for l in range(self.depth):
            in_dim = self.layer_dims[l]
            out_dim = self.layer_dims[l + 1]

            # shared parameters: [out_dim, in_dim], [out_dim]
            W_bar = nn.Parameter(torch.empty(out_dim, in_dim))
            b_bar = nn.Parameter(torch.empty(out_dim))

            # local residual parameters: [N, out_dim, in_dim], [N, out_dim]
            # dW = nn.Parameter(torch.empty(self.num_centers, out_dim, in_dim))
            # db = nn.Parameter(torch.empty(self.num_centers, out_dim))
            dW = nn.ParameterList([
                nn.Parameter(torch.empty(out_dim, in_dim))
                for _ in range(self.num_centers)
            ])

            db = nn.ParameterList([
                nn.Parameter(torch.empty(out_dim))
                for _ in range(self.num_centers)
            ])
            
            alpha = nn.Parameter(torch.ones(out_dim),requires_grad=True)
            self.alpha.append(alpha)
            self.W_shared.append(W_bar)
            self.b_shared.append(b_bar)
            self.dW_local.append(dW)
            self.db_local.append(db)

        self.out = nn.Linear(self.hidden_dim, 1, bias=True)

        self.reset_parameters()
        self._report_trainable()
        
    # def reset_parameters(self):
    #     for l in range(self.depth):
    #         nn.init.xavier_normal_(self.W_shared[l])
    #         nn.init.zeros_(self.b_shared[l])

    #         # local residual starts small, so model begins close to shared/global net
    #         # nn.init.normal_(self.dW_local[l], mean=0.0, std=0.02)
    #         # nn.init.zeros_(self.db_local[l])
    #         for i in range(self.num_centers):
    #             nn.init.normal_(self.dW_local[i], mean=0.0, std=0.02)
    #             nn.init.zeros_(self.db_local[i])



    #     nn.init.xavier_normal_(self.out.weight)
    #     nn.init.zeros_(self.out.bias)

    def reset_parameters(self):
        for l in range(self.depth):
            # shared/global parameters of layer l
            nn.init.xavier_normal_(self.W_shared[l])
            nn.init.zeros_(self.b_shared[l])

            # local residual parameters of layer l
            for i in range(self.num_centers):
                nn.init.normal_(self.dW_local[l][i], mean=0.0, std=0.02)
                nn.init.zeros_(self.db_local[l][i])

            # alpha of layer l
            nn.init.ones_(self.alpha[l])

        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    def _report_trainable(self):
        total = 0
        print("=== DeepSharedLocalBumpTanhNet Trainable parameters ===")
        for name, p in self.named_parameters():
            if p.requires_grad:
                n = p.numel()
                total += n
                print(f"{name}: {tuple(p.shape)} -> {n}")
        print(f"Total trainable params: {total}")
        print(f"Centers shape: {tuple(self.centers.shape)}")
        print(f"Radius shape : {tuple(self.radius.shape)}")
        print(f"Num centers  : {self.num_centers}")
        
    def _build_grid_centers(self, args) -> torch.Tensor:
            """
            Automatically generate centers on a Cartesian grid.

            For each dimension, sample uniformly from [center_min, center_max]
            with spacing center_step, then take the Cartesian product.

            Example:
                input_dim = 2
                center_min = -1
                center_max = 1
                center_step = 1
            gives centers:
                [-1,-1], [-1,0], [-1,1],
                [ 0,-1], [ 0,0], [ 0,1],
                [ 1,-1], [ 1,0], [ 1,1]
            """
            center_min = getattr(args, "center_min", -1.0)
            center_max = getattr(args, "center_max", 1.0)
            center_step = getattr(args, "center_step", 0.1) #0.05


            if center_step <= 0:
                raise ValueError("center_step must be positive.")
            if center_max < center_min:
                raise ValueError("center_max must be >= center_min.")

            # Include the right endpoint as much as possible
            axis = torch.arange(
                center_min,
                center_max + 0.5 * center_step,
                center_step
            )

            if axis.numel() == 0:
                raise ValueError("Generated axis is empty. Check center_min/center_max/center_step.")

            # Build d-dimensional Cartesian grid
            mesh = torch.meshgrid(*([axis] * self.input_dim), indexing="ij")

            # Stack into [N, d]
            centers = torch.stack([g.reshape(-1) for g in mesh], dim=-1)

            return centers
    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, d]
        diff = x[:, None, :] - self.centers[None, :, :]

        # [B, N]
        dist_sq = (diff ** 2).sum(dim=-1)

        r_sq = self.radius ** 2
        inside = dist_sq < r_sq

        # t = 1 - ||x - xi_i||^2 / r^2
        t = 1.0 - dist_sq / r_sq

        # 先构造一个安全版本，避免在 outside 位置除到 0 或负数
        safe_t = torch.where(inside, t, torch.ones_like(t))

        # 保持输出始终是 [B, N]，不做布尔索引取子张量
        phi_inside = torch.exp(-1.0 / safe_t)
        phi = torch.where(inside, phi_inside, torch.zeros_like(t))

        return phi

    def bump_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return normalized bump weights psi(x).

        Args:
            x: Tensor of shape [B, d]

        Returns:
            psi: Tensor of shape [B, N]
        """
        phi = self._phi(x)                           # [B, N]
        denom = phi.sum(dim=1, keepdim=True) + self.eps
        psi = phi / denom                            # [B, N]
        return psi

    def _forward_first_layer(self, x: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, d]
        psi: [B, N]
        out: [B, hidden_dim]
        """
        # Effective per-center weights:
        # W_eff[i] = W_shared + dW_local[i]
        
        alpha = self.alpha[0]                    # [out]
        alpha_w = alpha.view(1, -1, 1)                  # [1, out, 1]
        alpha_b = alpha.view(1, -1)                     # [1, out]
        dW = torch.stack(list(self.dW_local[0]), dim=0)   # [N, out, in]
        db = torch.stack(list(self.db_local[0]), dim=0)   # [N, out]
        W_eff = self.W_shared[0].unsqueeze(0) + alpha_w * dW[0]   # [N, out, in]
        b_eff = self.b_shared[0].unsqueeze(0) + alpha_b *db[0]   # [N, out]
        

        # W_eff = self.W_shared[0].unsqueeze(0) + alpha_w * self.dW_local[0]   # [N, out, in]
        # b_eff = self.b_shared[0].unsqueeze(0) + alpha_b * self.db_local[0]   # [N, out]

        # local_affine[b, n, m] = W_eff[n, m, :] · x[b, :] + b_eff[n, m]
        local_affine = torch.einsum("bd,nmd->bnm", x, W_eff)       # [B, N, m]
        local_affine = local_affine + b_eff.unsqueeze(0)           # [B, N, m]

        # aggregate over centers with psi
        z = torch.einsum("bn,bnm->bm", psi, local_affine)          # [B, m]

        h = torch.tanh(z)                                          # [B, m]
        return h

    def _forward_hidden_layer(self, h: torch.Tensor, psi: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Hidden layer l>=1:
            z = sum_i psi_i(x) * [ (W_bar + dW_i) h + b_bar + db_i ]

        h:         [B, in_dim]
        psi:       [B, N]
        layer_idx: 1, 2, ..., depth-1
        out:       [B, out_dim]
        """
        alpha = self.alpha[layer_idx]                    # [out]
        alpha_w = alpha.view(1, -1, 1)                  # [1, out, 1]
        alpha_b = alpha.view(1, -1)                     # [1, out]
        dW = torch.stack(list(self.dW_local[layer_idx]), dim=0)   # [N, out, in]
        db = torch.stack(list(self.db_local[layer_idx]), dim=0)   # [N, out]
        W_eff = self.W_shared[layer_idx].unsqueeze(0) + alpha_w * dW[layer_idx]   # [N, out, in]
        b_eff = self.b_shared[layer_idx].unsqueeze(0) + alpha_b *db[layer_idx]   # [N, out]
        # W_eff = self.W_shared[layer_idx].unsqueeze(0) + alpha_w * self.dW_local[layer_idx]   # [N, out, in]
        # b_eff = self.b_shared[layer_idx].unsqueeze(0) + alpha_b * self.db_local[layer_idx]   # [N, out]

        local_affine = torch.einsum("bp,nop->bno", h, W_eff)    # [B, N, out]
        local_affine = local_affine + b_eff.unsqueeze(0)        # [B, N, out]

        z = torch.einsum("bn,bno->bo", psi, local_affine)       # [B, out]
        h_next = torch.tanh(z)
        return h_next

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, d]
        y: [B, 1]
        """
        # bump weights are computed once from original input space
        psi = self.bump_weights(x)  # [B, N]

        # first layer uses x_hat^i = (x - xi_i) / r_i
        h = self._forward_first_layer(x, psi)

        # deeper layers use the same psi(x), but act on hidden features
        for l in range(1, self.depth):
            h = self._forward_hidden_layer(h, psi, l)

        y = self.out(h)  # [B, 1]
        return y
    
    def freeze_all_parameters(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_local_centers(self, center_indices):
        if isinstance(center_indices, int):
            center_indices = [center_indices]

        center_set = set(center_indices)

        for l in range(self.depth):
            for i in range(self.num_centers):
                flag = (i in center_set)
                self.dW_local[l][i].requires_grad_(flag)
                self.db_local[l][i].requires_grad_(flag)
        