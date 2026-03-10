# viz/time_plot_callback.py
import os
import time
import matplotlib.pyplot as plt
from viz.callbacks import Callback


class TimePlotCallback(Callback):
    def __init__(self, args, freq_dict={"adam": 5, "lbfgs": 1}):
        super().__init__()
        self.args = args
        self.save_dir = args.save_dir
        self.freq_dict = freq_dict
        self.freq = None

        os.makedirs(self.save_dir, exist_ok=True)

        self.times = []        # ← 横坐标
        self.history = {}      # loss history
        self.start_time = None

    def on_phase_begin(self, trainer, phase: str):
        if phase not in ("adam", "lbfgs", "proj_adam"):
            print(f"[Warning] Unknown phase '{phase}', using base freq.")

        self.freq = self.freq_dict.get(phase, 10)

        # 开始计时
        if self.start_time is None or trainer.iter_base==0:
            self.start_time = time.time()

    def on_iter_end(self, trainer, it: int, loss_dict: dict):

        # 当前时间
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)

        for k, v in loss_dict.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))

        if it % self.freq != 0:
            return

        self._save_plot(it)

    def on_train_end(self, trainer):
        if len(self.times) > 0:
            self._save_plot(self.times[-1])

    def _save_plot(self, it):

        plt.figure()

        for k, ys in self.history.items():
            plt.plot(self.times, ys, label=k)

        plt.xlabel("Time (seconds)")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title(f"Loss Curve (t={self.times[-1]:.1f}s)")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(self.save_dir, "time_loss_curve.png")
        plt.savefig(out_path, dpi=150)
        plt.close()