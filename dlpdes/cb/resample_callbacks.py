# viz/loss_plot_callback.py
import os
import matplotlib.pyplot as plt
from cb.callbacks import Callback
from data.rard2d import RARDResampler2D

class ResamplePlotCallback(Callback):
    def __init__(self, args,  freq_dict={"adam":500,"lbfgs":500,"lm":100}):
        super().__init__()
        self.args = args
        self.freq_dict = freq_dict
        self.freq = None

    def on_iter_end(self, trainer, it: int, loss_dict: dict):
        # loss_dict should already be pure floats in your trainer; if not, it still works for float-like.
        if it % self.freq == 0:
            rard=RARDResampler2D(
            trainer.eq,
            trainer.args.device
            )
            data,info =rard.resample(trainer.model,trainer.data)
            #resample
            trainer.data=data
            print(
            f"old_n={info['old_n']} "
            f"new_n={info['new_n']} "
            f"added={info['added_n']} "
            f"k={info['k']:.2f} "
            f"c={info['c']:.2f} "
            f"res_max={info.get('res_abs_max', 0.0):.3e}"
        )
        

        
    def on_phase_begin(self, trainer, phase: str):
        if phase not in ("adam", "lbfgs","proj_adam", "lm"):
            print(f"[Warning] Unknown phase '{phase}', using base freq.")
        self.freq = self.freq_dict.get(phase, self._base_freq)
