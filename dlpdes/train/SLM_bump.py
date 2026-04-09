import math
import numpy as np
import torch 
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from  torch.func import functional_call, vmap, jacfwd, jacrev
from .utils import *
import gc
# The stochastic_Levenberg-Marquardt method for bumpmlp

# remove beta training function

from collections import OrderedDict

import torch
def Jacobian_function(x, eq, model, index=None):


    params = dict(model.named_parameters())
    # total_numel = sum(p.numel() for p in params.values())
    # print("total parameters:", total_numel)
    #（）
    named_params = list(model.named_parameters())
    # param_groups = group_named_parameters_by_size(
    #     named_params,xw
    #     max_group_numel=5_000,
    # )
    
    buffers = dict(model.named_buffers())

    def residual_func(params, x):
        if isinstance(x, dict):
            x_in = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in x.items()}
        else:
            x_in = x.unsqueeze(0) if x.ndim == 1 else x
        model_fn = lambda x: functional_call(model, {**params, **buffers}, (x,))
        loss_dict = eq.compute_loss(model_fn, x_in,mode="jacrev")

        r = loss_dict["residuals"]["all"]      # [n_res]
        aux = (
            loss_dict["loss"]["total"],
            loss_dict,
        )
        return r, aux

    J_dict, (total_loss, loss_dict) = jacfwd(
        residual_func,
        argnums=0,
        has_aux=True,
    )(params, x)

    J_blocks = []
    for name, p in params.items():
        Ji = J_dict[name]                      # [n_res, *p.shape]
        J_blocks.append(Ji.reshape(Ji.shape[0], -1))

    J = torch.cat(J_blocks, dim=1)   

    r_full = loss_dict["residuals"]["all"].detach()
    total_loss = total_loss.detach()
    # J_F = J[:, index]
    
    del J_dict
    del J_blocks 
    gc.collect()
    torch.cuda.empty_cache()

    # 5️⃣ 返回 Jacobian, residual, total loss
    return J, r_full, total_loss,loss_dict


    
    
def levenberg_marquardt(trainer,data):
    epochs=getattr(trainer.args, 'lm_epochs', 1500)
    _set_phase(trainer, "lm")

    for cb in trainer.callbacks:
        cb.on_train_begin(trainer)
        
    lm_beta_train=getattr(trainer.args, 'lm_beta_train', False) # whether to update beta during training
    lm_miu=getattr(trainer.args, 'lm_miu', 1e-3) # damping factor

    N_params = parameters_to_vector(trainer.model.parameters()).numel() # N = number of parameters
    lm_gama=getattr(trainer.args, 'lm_gama', 2.0) # factor for adjusting miu ; gama>1
    lm_yita1=getattr(trainer.args, 'lm_yita1', 1e-16) # threshold for decreasing miu
    lm_yita2=getattr(trainer.args, 'lm_yita2', 1e-16) # threshold for increasing miu
    lm_min_miu=getattr(trainer.args, 'lm_min_miu', 1e-32) # min miu
    lm_max_miu=getattr(trainer.args, 'lm_max_miu', 1e100) # max miu to prevent overflow
    train_tol= getattr(trainer.args, 'lm_train_tol', 1e-5) # training stopping criterion based on gradient norm
    lm_beta= N_params
    lm_index = np.random.choice(N_params, lm_beta, replace=False)  # random subset of parameters
    it=trainer.iter_base
    
    
    for epoch in range(epochs):
        data=refresh_batch(data)  # 刷新 batch 中的张量，使其成为叶子节点，允许反向传播
        trainer.model.zero_grad(set_to_none=True)
        it += 1
        param_old=parameters_to_vector(trainer.model.parameters()) # save current parameters for potential rollback
        # shape: (N_batch,N_params_subset),(N_batch,N_params) (N,), scalar,dict
        J_F,r,loss_old,loss_dict= Jacobian_function(data, trainer.eq,trainer.model, index=lm_index)   
        r= r.view(-1,1)  # (N,1)
        with torch.no_grad():
            JTr_F = torch.matmul(J_F.T, r)  # (subset Jacobian^T) @ (residual) -> [k, 1]
            JTJ_F = torch.matmul(J_F.T, J_F)  # (subset Jacobian^T) @ (subset Jacobian) -> [k, k]
            A = JTJ_F + lm_miu * torch.eye(JTJ_F.shape[0], device=JTJ_F.device, dtype=JTJ_F.dtype) 
            # print("condition number:", torch.linalg.cond(A))
            delta = torch.linalg.solve(A, -JTr_F)  # theta delta (subset) h= (J^T J + miu I)^{-1} @ (-J^T r) -> [k, 1]
            
        #reconstruct full delta
            # delta_full = torch.zeros_like(param_old,device=param_old.device)
            # delta_full[lm_index] = delta.reshape(-1)
            
            delta=delta.reshape(-1)
            param_new=param_old +delta
            
            #
            
            #
            vector_to_parameters(param_new, trainer.model.parameters())
        
        
        #caculate rho ------------------
        loss_new=trainer.eq.compute_loss(trainer.model, data,mode="jacrev")['loss']['total'].item()
        rho_denom=-JTr_F.T @ delta- 0.5*delta.T@A@ delta
        # print(f"rho_denom={rho_denom.item():.3e}")
        rho_numer=loss_old-loss_new
        lm_rho=(rho_numer/rho_denom).item()
        gk_norm_F=torch.norm(JTr_F, p=2)
        
        
        #update strategy----------------------
        if lm_rho>=lm_yita1 and gk_norm_F**2>=lm_yita2/lm_miu:
            # print("success") #test
            trainer.param_delta.setdefault("update", []).append(delta.detach())
            lm_miu=max(lm_miu/lm_gama,lm_min_miu)
        else:
            vector_to_parameters(param_old, trainer.model.parameters()) # revert to old params
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
            
        del J_F, r, loss_old, loss_dict, JTr_F, JTJ_F, A, delta,  param_new 
        gc.collect()
        torch.cuda.empty_cache()
        
    ### out loop
            
    for cb in trainer.callbacks:
        cb.on_train_end(trainer)

        

        
        
        
    







