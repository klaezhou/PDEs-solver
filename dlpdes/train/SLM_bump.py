import math
import numpy as np
import torch 
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from  torch.func import functional_call, vmap, jacfwd, jacrev
from .utils import *
import gc
# The Levenberg-Marquardt method

# remove beta training function

from collections import OrderedDict

import torch
import torch
from torch.func import jacrev, functional_call

def _iter_slices(n, chunk_size):
    for s in range(0, n, chunk_size):
        yield slice(s, min(s + chunk_size, n))


def _flatten_jdict(J_dict, params, m):
    blocks = []
    for name, p in params.items():
        Ji = J_dict[name]              # [m, *p.shape]
        blocks.append(Ji.reshape(m, -1))
    return torch.cat(blocks, dim=1)


def build_normal_eq_chunked(data,eq,model,model_old,f_chunk=100,b_chunk=100,args=None):
    """
    返回:
        JTJ: [nparam, nparam]
        JTr: [nparam]
        lsq_loss: 0.5 * ||r||^2
        loss_dict_full: 仅用于监控（不是 LM 的完整目标）
    """
    anchor_weight = getattr(args, "anchor_weight", 1000.0)

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())

    if len(params) == 0:
        raise RuntimeError("No trainable parameters found.")

    first_p = next(iter(params.values()))
    device = first_p.device
    dtype = first_p.dtype
    nparam = sum(p.numel() for p in params.values())

    JTJ = torch.zeros(nparam, nparam, device=device, dtype=dtype)
    JTr = torch.zeros(nparam, device=device, dtype=dtype)
    rr  = torch.zeros((), device=device, dtype=dtype)

    # 旧模型只用来给 anchor 提供 target，先预计算，后面就不必一直带着 model_old 了
    model_old.eval()
    with torch.no_grad():
        u_old_f = model_old(data["X_f"]).detach()
        u_old_b = model_old(data["X_b"]).detach()
        u_old=model_old(data["LM_b"]).detach()
    anchor_dim_f = u_old_f.numel()
    anchor_dim_b = u_old_b.numel()
    anchor_dim_lm=u_old.numel()

    empty_X_f = data["X_f"][:0]
    empty_f_f = data["f_f"][:0]
    empty_X_b = data["X_b"][:0]
    empty_g_b = data["g_b"][:0]
    empty_lm_b = data["LM_b"][:0]

    def make_model_fn(curr_params):
        return lambda x: functional_call(model, {**curr_params, **buffers}, (x,))

    def accumulate_one_residual_fn(residual_fn, batch_obj):
        nonlocal JTJ, JTr, rr

        # jacrev 外层支持 chunk_size；chunk_size=1 等价于逐行 for-loop，更稳
        J_dict, r = jacrev(
            residual_fn,
            argnums=0,
            has_aux=True,
        )(params, batch_obj)

        r = r.detach().reshape(-1)
        if r.numel() == 0:
            return

        J = _flatten_jdict(J_dict, params, r.numel())   # [m_chunk, nparam]
        with torch.no_grad():
            JTJ.add_(J.T @ J)
            JTr.add_(J.T @ r)
            rr.add_(r @ r)

        del J_dict, J, r

    # -------- 1) interior / PDE residual on X_f chunks --------
    def residual_f(curr_params, batch_f):
        model_fn = make_model_fn(curr_params)
        batch = {
            "X_f": batch_f["X_f"],
            "f_f": batch_f["f_f"],
            "X_b": empty_X_b,
            "g_b": empty_g_b,
            "LM_b":empty_lm_b,
        }
        # loss_dict = eq.compute_loss(model_fn, batch, mode="jacrev")

        loss_dict = eq.compute_loss(model_fn, batch, mode="jacrev")

        
        r = loss_dict["residuals"]["all"].reshape(-1)
        r= r*math.sqrt(r.numel())/math.sqrt(anchor_dim_f+anchor_dim_b)
        return r, r.detach()

    for sl in _iter_slices(data["X_f"].shape[0], f_chunk):
        batch_f = {
            "X_f": data["X_f"][sl],
            "f_f": data["f_f"][sl],
        }
        accumulate_one_residual_fn(residual_f, batch_f)


    # -------- 2) boundary residual on X_b chunks --------
    def residual_b(curr_params, batch_b):
        model_fn = make_model_fn(curr_params)
        batch = {
            "X_f": empty_X_f,
            "f_f": empty_f_f,
            "X_b": batch_b["X_b"],
            "g_b": batch_b["g_b"],
            "LM_b":empty_lm_b,
        }
        loss_dict = eq.compute_loss(model_fn, batch, mode="jacrev")
        r = loss_dict["residuals"]["all"].reshape(-1)
        r=r*math.sqrt(r.numel())/math.sqrt(anchor_dim_b+anchor_dim_f)
        return r, r.detach()

    for sl in _iter_slices(data["X_b"].shape[0], b_chunk):
        batch_b = {
            "X_b": data["X_b"][sl],
            "g_b": data["g_b"][sl],
        }
        accumulate_one_residual_fn(residual_b, batch_b)


    # -------- 4) anchor on X_b chunks --------
    def residual_anchor_b(curr_params, batch_ba):
        model_fn = make_model_fn(curr_params)
        pred = model_fn(batch_ba["LM_b"])
        r = anchor_weight * (batch_ba["u_old"] - pred).reshape(-1)
        r=r/math.sqrt(anchor_dim_lm)
        return r, r.detach()

    for sl in _iter_slices(data["LM_b"].shape[0], b_chunk):
        batch_ba = {
            "LM_b": data["LM_b"][sl],
            "u_old": u_old[sl],
        }
        accumulate_one_residual_fn(residual_anchor_b, batch_ba)

    # 监控用：单独算一次当前 full loss_dict
    # 注意：这不是 LM 真正优化的完整 least-squares 目标，因为里面通常不含 anchor
    full_model_fn = make_model_fn(params)
    loss_dict_full = eq.compute_loss(full_model_fn, data, mode="jacrev")

    lsq_loss = 0.5 * rr.detach()
    return JTJ, JTr, lsq_loss, loss_dict_full



@torch.no_grad()
def compute_lm_loss(data, eq, model, model_old):
    anchor_weight = getattr(eq, "anchor_weight", 1.0)
    model.eval()
    model_old.eval()
    with torch.no_grad():
        u_old_f = model_old(data["X_f"]).detach()
        u_old_b = model_old(data["X_b"]).detach()
        u_old= model_old(data["LM_b"]).detach()
    anchor_dim_f = u_old_f.numel()
    anchor_dim_b = u_old_b.numel()

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    total_rr = torch.zeros((), device=device, dtype=dtype)

    # 1) PDE/BC residual part
    loss_dict = eq.compute_loss(model, data, mode="jacrev")
    r_main = loss_dict["residuals"]["all"].reshape(-1)
    r_main = r_main* math.sqrt(r_main.numel())/math.sqrt(anchor_dim_f+anchor_dim_b)
    total_rr += r_main @ r_main

    # 3) anchor on X_b (full batch)
    LM_b = data["LM_b"]
    if LM_b.numel() > 0:
        r_anchor_b = anchor_weight * (model(LM_b) - u_old)
        r_anchor_b = r_anchor_b.reshape(-1)
        r_anchor_b = r_anchor_b / math.sqrt(r_anchor_b.numel())
        total_rr += r_anchor_b @ r_anchor_b

    return 0.5 * total_rr.item()
def Slevenberg_marquardt(trainer,data,model_old):
    # use model_old to constraint lm training process
    model_old.eval()
    epochs=getattr(trainer.args, 'lm_epochs', 1500)
    _set_phase(trainer, "lm")

    for cb in trainer.callbacks:
        cb.on_train_begin(trainer)
        
    lm_beta_train=getattr(trainer.args, 'lm_beta_train', False) # whether to update beta during training
    lm_miu=getattr(trainer.args, 'lm_miu', 1e-2) # damping factor
    
    # N_params = parameters_to_vector(trainer.model.parameters()).numel() # N = number of parameters
    N_params = parameters_to_vector([p for p in trainer.model.parameters() if p.requires_grad]).numel()
    lm_gama=getattr(trainer.args, 'lm_gama', 4.0) # factor for adjusting miu ; gama>1
    lm_yita1=getattr(trainer.args, 'lm_yita1', 1e-16) # threshold for decreasing miu
    lm_yita2=getattr(trainer.args, 'lm_yita2', 1e-16) # threshold for increasing miu
    lm_min_miu=getattr(trainer.args, 'lm_min_miu', 1e-32) # min miu
    lm_max_miu=getattr(trainer.args, 'lm_max_miu', 1e100) # max miu to prevent overflow
    train_tol= getattr(trainer.args, 'lm_train_tol', 1e-13) # training stopping criterion based on gradient norm
    lm_beta= N_params
    lm_index = np.random.choice(N_params, lm_beta, replace=False)  # random subset of parameters
    it=trainer.iter_base
    
    
    for epoch in range(epochs):
        data=refresh_batch(data)  # 刷新 batch 中的张量，使其成为叶子节点，允许反向传播
        trainer.model.zero_grad(set_to_none=True)
        it += 1
        # param_old=parameters_to_vector(trainer.model.parameters()) # save current parameters for potential rollback
        param_old=parameters_to_vector([p for p in trainer.model.parameters() if p.requires_grad])
        # shape: (N_batch,N_params_subset),(N_batch,N_params) (N,), scalar,dict
        JTJ_F,JTr_F,loss_old,loss_dict=build_normal_eq_chunked(data, trainer.eq,trainer.model,model_old,args=trainer.args)
        # loss_new = compute_lm_loss(data, trainer.eq, trainer.model, model_old)
        # print("[LM] epoch: %d, loss_old: %.5e, loss_new: %.5e" % (epoch, loss_old, loss_new))
        # r= r.view(-1,1)  # (N,1)
        with torch.no_grad():
            # JTr_F = torch.matmul(J_F.T, r)  # (subset Jacobian^T) @ (residual) -> [k, 1]
            # JTJ_F = torch.matmul(J_F.T, J_F)  # (subset Jacobian^T) @ (subset Jacobian) -> [k, k]
            #torch.diag_embed(JTJ_F.diagonal()) torch.eye(JTJ_F.shape[0], device=JTJ_F.device, dtype=JTJ_F.dtype) 
            A = JTJ_F + lm_miu * torch.diag_embed(JTJ_F.diagonal())+ 1e-8* torch.eye(JTJ_F.shape[0], device=JTJ_F.device, dtype=JTJ_F.dtype)
            # print("condition number:", torch.linalg.cond(A))
            try :
                delta = torch.linalg.solve(A, -JTr_F)  # theta delta (subset) h= (J^T J + miu I)^{-1} @ (-J^T r) -> [k, 1]
            except RuntimeError as e:
                print(f"[LM] torch.linalg.solve failed: {e}")
                # delta = torch.linalg.lstsq(A, -JTr_F).solution
                break
            delta=delta.reshape(-1)
            param_new=param_old +delta
            
            # vector_to_parameters(param_new, trainer.model.parameters())
            vector_to_parameters(param_new, [p for p in trainer.model.parameters() if p.requires_grad])
        
        
        #caculate rho ------------------
        loss_new=compute_lm_loss(data, trainer.eq, trainer.model, model_old)
        rho_denom=-JTr_F.T @ delta- 0.5*delta.T@A@ delta
        rho_numer=loss_old-loss_new
        # print(f"rho_numer={loss_old}, rho_denom={loss_new}")
        lm_rho=(rho_numer/rho_denom).item()
        gk_norm_F=torch.norm(JTr_F, p=2)
        
        if gk_norm_F**2<train_tol:
            print("converged")
            break
        #update strategy----------------------
        if lm_rho>=lm_yita1 and gk_norm_F**2>=lm_yita2/lm_miu:
            # print("success") #test
            trainer.param_delta.setdefault("update", []).append(delta.detach())
            lm_miu=max(lm_miu/lm_gama,lm_min_miu)
        else:
            print(f"miu",lm_miu)
            vector_to_parameters(param_old, [p for p in trainer.model.parameters() if p.requires_grad])
            lm_miu=min(lm_miu*lm_gama,lm_max_miu)
        if lm_beta_train:
            JTr= torch.matmul(J.T, r)  # (full Jacobian^T) @ (residual) -> [N, 1]
            gk_norm=torch.norm(JTr, p=2)
            lm_beta = int(np.round(N_params* torch.sqrt(1 - 1 / (2 * gk_norm**4 * lm_miu**2)).item())) \
                    if 1 / (2 * gk_norm**4 * lm_miu**2) <= 1 \
                    else int(np.round(1 / 2 * N_params))
            del JTr, gk_norm
        if it % trainer.log_freq == 0:
            trainer._print_log(it, loss_dict)
            print(f"[LM] iter={it}, beta={lm_beta}, gk_norm={gk_norm_F.item():.3e},lm_miu={lm_miu:.2e}")
            

        for cb in trainer.callbacks:
            cb.on_iter_end(trainer, it, loss_dict)
            
        del  loss_old, loss_dict, JTr_F, JTJ_F, A, delta,  param_new  #J_F, r,
        gc.collect()
        torch.cuda.empty_cache()
        
    ### out loop
            
    for cb in trainer.callbacks:
        cb.on_train_end(trainer)

        

        
        
        
    







