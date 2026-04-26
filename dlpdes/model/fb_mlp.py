import torch
import torch.nn as nn


class TanhMLP(nn.Module):
    """
    Standard tanh MLP.

    If depth = 0:
        x -> Linear(input_dim, out_dim)

    If depth >= 1:
        x -> [Linear + tanh] * depth -> Linear -> out
    """

    def __init__(self, input_dim, hidden_dim, depth, out_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.out_dim = out_dim

        self.layers = nn.ModuleList()

        in_dim = input_dim
        for _ in range(depth):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.out = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        y = self.out(h)
        return y


class FBTanhNet(nn.Module):
    """
    Additive local-expert model with bump-function gating.

    Model:
        u(x) = u_shared(x) + sum_i psi_i(x) * beta_i * e_i(x)

    where
        - u_shared(x): global/shared MLP
        - e_i(x): local expert MLP for center i
        - beta_i: trainable scalar for expert i, initialized to 0
        - psi_i(x): bump weights, optionally normalized

    Compared with the old "MoE inside activation" form, this is an additive
    interface: local experts are separated from the shared/global network.
    """

    def __init__(self, args):
        super().__init__()

        # ===== basic hyperparameters =====
        self.input_dim = getattr(args, "input_dim", 2)
        self.hidden_dim = getattr(args, "hidden_dim", 50)
        self.depth = getattr(args, "bump_depth", 1)

        # optional separate hyperparameters
        self.shared_hidden_dim = getattr(args, "shared_hidden_dim", self.hidden_dim)
        self.shared_depth = getattr(args, "shared_depth", self.depth)
        self.local_hidden_dim = getattr(args, "local_hidden_dim", self.hidden_dim)
        self.local_depth = getattr(args, "local_depth", self.depth)

        self.eps = getattr(args, "eps", 1e-12)
        self.trainable_centers = getattr(args, "trainable_centers", False)
        self.normalize_bump = getattr(args, "normalize_bump", True)

        radius = getattr(args, "radius", 0.3)
        self.local_scale_init = getattr(args, "local_scale_init", 0.0)

        # ===== centers =====
        centers = self._build_grid_centers(args)
        self.num_centers = centers.shape[0]

        assert centers.shape == (self.num_centers, self.input_dim), (
            f"Expected centers shape [{self.num_centers}, {self.input_dim}], "
            f"but got {tuple(centers.shape)}"
        )
        assert radius > 0.0, "radius must be positive"

        if self.trainable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

        self.register_buffer("radius", torch.tensor(float(radius), dtype=torch.float32))

        # ===== shared/global network =====
        self.shared_net = TanhMLP(
            input_dim=self.input_dim,
            hidden_dim=self.shared_hidden_dim,
            depth=self.shared_depth,
            out_dim=1,
        )

        # ===== local experts =====
        self.local_experts = nn.ModuleList([
            TanhMLP(
                input_dim=self.input_dim,
                hidden_dim=self.local_hidden_dim,
                depth=self.local_depth,
                out_dim=1,
            )
            for _ in range(self.num_centers)
        ])

        # ===== local residual scales beta_i =====
        # important: init to zero, so initial model is exactly the shared model
        self.local_scale = nn.ParameterList([
            nn.Parameter(torch.tensor([self.local_scale_init], dtype=torch.float32))
            for _ in range(self.num_centers)
        ])

        self.reset_parameters()
        self._report_trainable()

    def reset_parameters(self):
        # shared/global network
        self.shared_net.reset_parameters()

        # local experts: random init is okay because beta_i starts from 0
        for expert in self.local_experts:
            expert.reset_parameters()

        # local scales
        for beta in self.local_scale:
            with torch.no_grad():
                beta.fill_(self.local_scale_init)

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
            print(f"{k:<15} -> {v}")
        print(f"Total trainable params: {total}")
        print(f"Num centers            : {self.num_centers}")

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
            center_min,
            center_max + 0.5 * center_step,
            center_step
        )

        if axis.numel() == 0:
            raise ValueError("Generated axis is empty. Check center_min/center_max/center_step.")

        mesh = torch.meshgrid(*([axis] * self.input_dim), indexing="ij")
        centers = torch.stack([g.reshape(-1) for g in mesh], dim=-1)
        return centers.float()

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compactly supported radial bump.

        x:   [B, d]
        out: [B, N]
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
        Return bump weights psi(x).

        If normalize_bump = True:
            psi_i = phi_i / sum_j phi_j

        else:
            psi_i = phi_i
        """
        phi = self._phi(x)  # [B, N]

        if self.normalize_bump:
            denom = phi.sum(dim=1, keepdim=True) + self.eps
            psi = phi / denom
            return psi

        return phi

    def shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shared/global branch output.

        x: [B, d]
        y: [B, 1]
        """
        return self.shared_net(x)

    def shared_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Penultimate features of the shared/global network.
        """
        return self.shared_net.forward_features(x)

    def local_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all local experts.

        x: [B, d]
        return:
            local_vals: [B, N], where local_vals[:, i] = beta_i * e_i(x)
        """
        vals = []
        for i, expert in enumerate(self.local_experts):
            yi = expert(x)                       # [B, 1]
            yi = yi * self.local_scale[i]        # [B, 1]
            vals.append(yi)

        local_vals = torch.cat(vals, dim=1)      # [B, N]
        return local_vals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, d]
        y: [B, 1]
        """
        psi = self.bump_weights(x)               # [B, N]
        y_shared = self.shared_forward(x)        # [B, 1]
        y_local = self.local_outputs(x)          # [B, N]

        y = y_shared + (psi * y_local).sum(dim=1, keepdim=True)
        return y

    # ------------------------------------------------------------------
    # freezing / unfreezing utilities
    # ------------------------------------------------------------------

    def freeze_all_parameters(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_local_centers(self, center_indices, with_scale=True):
        """
        Unfreeze selected local experts (and optionally their beta_i).

        center_indices:
            int or list[int]
        """
        if isinstance(center_indices, int):
            center_indices = [center_indices]

        center_set = set(center_indices)

        for i in range(self.num_centers):
            flag = (i in center_set)

            for p in self.local_experts[i].parameters():
                p.requires_grad_(flag)

            if with_scale:
                self.local_scale[i].requires_grad_(flag)
            else:
                self.local_scale[i].requires_grad_(False)

    def unfreeze_shared(self, layer_indices=None, with_bias=True, with_out=True):
        """
        Unfreeze shared/global network.

        layer_indices:
            - None: unfreeze all hidden layers
            - int or list[int]: unfreeze only selected hidden layers

        with_bias:
            whether to unfreeze bias of hidden layers

        with_out:
            whether to unfreeze the output layer of shared_net
        """
        num_hidden_layers = len(self.shared_net.layers)

        if layer_indices is None:
            layer_indices = list(range(num_hidden_layers))
        elif isinstance(layer_indices, int):
            layer_indices = [layer_indices]

        layer_set = set(layer_indices)

        for l, layer in enumerate(self.shared_net.layers):
            flag = (l in layer_set)
            layer.weight.requires_grad_(flag)

            if layer.bias is not None:
                layer.bias.requires_grad_(flag and with_bias)

        self.shared_net.out.weight.requires_grad_(with_out)
        if self.shared_net.out.bias is not None:
            self.shared_net.out.bias.requires_grad_(with_out)

    def unfreeze_centers(self, flag=True):
        """
        Unfreeze/freeze center coordinates if centers are trainable.
        """
        if isinstance(self.centers, nn.Parameter):
            self.centers.requires_grad_(flag)

    # ------------------------------------------------------------------
    # optional helpers
    # ------------------------------------------------------------------

    def get_local_scale_tensor(self) -> torch.Tensor:
        """
        Return beta_i as a tensor of shape [N].
        """
        return torch.cat([p.detach().view(1) for p in self.local_scale], dim=0)

    def get_raw_phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return raw unnormalized bump values phi(x).
        """
        return self._phi(x)


def fb_mlp_penultimate_getter(model, x):
    """
    Return the penultimate shared feature of the additive model.

    Parameters
    ----------
    model : LocalBumpTanhNet
    x     : [B, input_dim]

    Returns
    -------
    h : [B, shared_hidden_dim]  if shared_depth >= 1
        [B, input_dim]          if shared_depth = 0
    """
    return model.shared_features(x)


def bump_local_outputs_getter(model, x):
    """
    Useful debug helper.

    Returns
    -------
    psi      : [B, N]
    local_y  : [B, N]   (already includes beta_i)
    """
    psi = model.bump_weights(x)
    local_y = model.local_outputs(x)
    return psi, local_y