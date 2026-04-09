import torch


class RARDResampler2D:
    """
    2D Residual-based Adaptive Refinement with Distribution (RAR-D)

    For current repo structure:
      - batch contains: X_f, X_b, f_f, g_b
      - only interior points X_f are updated
      - boundary points X_b stay fixed
      - append-only: X_f <- concat(X_f, X_new)

    PDF:
        p(x) ∝ eps(x)^k / E[eps(x)^k] + c
    where
        eps(x) = |PDE residual at x|
    """

    def __init__(
        self,
        eq,
        device,
        lowx=-1.0,
        highx=1.0,
        lowy=-1.0,
        highy=1.0,
        candidate_size=10000,   # |S0|
        add_size=1000,          # m
        k=1.0,
        c=1.0,
        eps_floor=1e-12,
        deduplicate=False,
        max_points=5000,        # optional hard cap for total interior points
        drop_mode="low_residual" # "low_residual" "random"
    ):
        self.eq = eq
        self.device = device

        self.lowx = lowx
        self.highx = highx
        self.lowy = lowy
        self.highy = highy

        self.candidate_size = candidate_size
        self.add_size = add_size
        self.k = k
        self.c = c
        self.eps_floor = eps_floor
        self.deduplicate = deduplicate
        self.max_points = max_points
        self.drop_mode= drop_mode

    # --------------------------------------------------
    # basic utilities
    # --------------------------------------------------
    def sample_interior_candidates(self, n, dtype=torch.float32):
        x = self.lowx + (self.highx - self.lowx) * torch.rand(
            n, 1, device=self.device, dtype=dtype
        )
        y = self.lowy + (self.highy - self.lowy) * torch.rand(
            n, 1, device=self.device, dtype=dtype
        )
        return torch.cat([x, y], dim=1)   # [n, 2]

    def _build_batch_with_new_Xf(self, batch, X_f_new):
        X_f_new = X_f_new.detach().clone().to(self.device)
        X_f_new = X_f_new.requires_grad_(True)

        new_batch = dict(batch)
        new_batch["X_f"] = X_f_new
        new_batch["f_f"] = self.eq.f(X_f_new)
        return new_batch

    def _extract_interior_residual(self, loss_dict, n_f):
        """
        In current poisson.py:
            r = cat([r_f.flatten(), r_b.flatten()])
        so first n_f entries are interior PDE residuals.
        """
        r_all = loss_dict["residuals"]["all"].reshape(-1)
        r_f = r_all[:n_f]
        return r_f

    @torch.no_grad()
    def _candidate_residual_abs(self, model, batch, X_cand):
        """
        Evaluate |r_f| on candidate interior points.
        No training update here; just score computation.
        """
        temp_batch = self._build_batch_with_new_Xf(batch, X_cand)

        was_training = model.training
        model.eval()
        loss_dict = self.eq.compute_loss(model, temp_batch, mode="jacrev")
        if was_training:
            model.train()

        r_f = self._extract_interior_residual(loss_dict, X_cand.shape[0])
        eps = torch.abs(r_f).detach().reshape(-1)
        eps = torch.clamp(eps, min=self.eps_floor)
        return eps

    def _build_sampling_prob(self, eps):
        """
        eps: [M], absolute PDE residual on candidate pool

        p_i ∝ eps_i^k / mean(eps^k) + c
        """
        score = eps.pow(self.k)
        mean_score = torch.clamp(score.mean(), min=self.eps_floor)

        weight = score / mean_score + self.c
        weight = torch.clamp(weight, min=self.eps_floor)

        prob = weight / torch.clamp(weight.sum(), min=self.eps_floor)
        return prob, weight

    def _maybe_unique(self, X):
        if not self.deduplicate:
            return X
        return torch.unique(X, dim=0)

    # --------------------------------------------------
    # core RAR-D step
    # --------------------------------------------------
    def sample_new_points(self, model, batch, m=None):
        """
        1) sample dense candidate pool S0
        2) compute residual-based PDF on S0
        3) sample m points from S0 according to that PDF
        """
        if m is None:
            m = self.add_size

        dtype = batch["X_f"].dtype
        X_cand = self.sample_interior_candidates(self.candidate_size, dtype=dtype)  # [M,2]

        eps = self._candidate_residual_abs(model, batch, X_cand)                    # [M]
        prob, weight = self._build_sampling_prob(eps)                                # [M]

        m = min(m, X_cand.shape[0])
        idx = torch.multinomial(prob, num_samples=m, replacement=False)
        X_new = X_cand[idx]

        info = {
            "candidate_size": int(X_cand.shape[0]),
            "selected_size": int(X_new.shape[0]),
            "res_abs_mean": float(eps.mean().item()),
            "res_abs_max": float(eps.max().item()),
            "weight_mean": float(weight.mean().item()),
            "weight_max": float(weight.max().item()),
            "k": float(self.k),
            "c": float(self.c),
        }
        return X_new, info

    def _drop_old_points(self, model, batch, X_old, n_drop):
        """
        Drop n_drop old interior points.
        """
        n_old = X_old.shape[0]
        n_drop = min(n_drop, n_old - 1)   # 至少保留 1 个点

        if n_drop <= 0:
            return X_old, 0

        if self.drop_mode == "random":
            perm = torch.randperm(n_old, device=X_old.device)
            keep_idx = perm[n_drop:]

        elif self.drop_mode == "low_residual":
            eps_old = self._candidate_residual_abs(model, batch, X_old)
            keep_n = n_old - n_drop
            keep_idx = torch.topk(eps_old, k=keep_n, largest=True).indices
            # 保留残差大的旧点，删掉残差小的旧点

        else:
            raise ValueError(f"Unknown drop_mode: {self.drop_mode}")

        return X_old[keep_idx], n_drop

    def resample(self, model, batch, m=None):
        """
        Append-only RAR-D:
            X_f <- concat(X_f, X_new)
        """
        X_old = batch["X_f"].detach()
        old_n = X_old.shape[0]
        if m is None:
            m = self.add_size

        if self.max_points is not None and old_n + m > self.max_points:
            self.prune_size=m
            X_old, removed_n = self._drop_old_points(
                model, batch, X_old, self.prune_size
            )
            batch = self._build_batch_with_new_Xf(batch, X_old)
            old_n = X_old.shape[0]


        if m <= 0:
            info = {
                "mode": "rard_append_only",
                "old_n": int(old_n),
                "new_n": int(old_n),
                "added_n": 0,
                "stopped": True,
                "reason": "no_room_for_new_points",
                "k": float(self.k),
                "c": float(self.c),
            }
            return batch, info

        X_new, info = self.sample_new_points(model, batch, m=m)
        X_next = torch.cat([X_old, X_new], dim=0)
        X_next = self._maybe_unique(X_next)

        new_batch = self._build_batch_with_new_Xf(batch, X_next)

        info.update({
            "mode": "rard_append_only",
            "old_n": int(old_n),
            "new_n": int(X_next.shape[0]),
            "added_n": int(X_next.shape[0] - old_n),
            "stopped": False,
        })
        return new_batch, info