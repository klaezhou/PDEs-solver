__all__ = ["refresh_batch", "_set_phase"]
def refresh_batch(batch):
    """
    将固定 batch 重新包装成叶子张量，每次训练 step 都可以安全 backward
    batch: dict，包含 "X_f", "X_b", "f_f", "g_b"
    return: 新的 batch dict
    """
    for key in batch:
            batch[key] = batch[key].detach().clone().requires_grad_(True)
    return batch



def _set_phase(trainer, phase: str):
        trainer.phase = phase
        for cb in trainer.callbacks:
                cb.on_phase_begin(trainer, phase)
                
        trainer.log_freq=trainer.args.log_freq.get(phase, 100)
        
