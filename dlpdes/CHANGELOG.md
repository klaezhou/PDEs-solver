# Current structure (`dlpdes/`)

## Overview

`dlpdes/` has now evolved into a relatively complete PINN experimental framework. Its overall structure is organized by layers: **equation definition → data generation → model construction → training orchestration → metric analysis → visualization callbacks**. Each script first parses arguments, constructs the equation object and feature extractor through factory functions, then assembles the `Pipeline` together with multiple callbacks.

## Top-level layout

- `ac_run.py`  
  Entry point for the Allen–Cahn task. It defines spatio-temporal sampling, PDE/BC/IC weights, model hyperparameters, training frequencies, and rank-analysis settings, while wiring together `ErrorPlotCallback`, `LossPlotCallback`, `CheckpointCallback`, and `RankCallback`. The training flow is `train_adam() → reset_model() → reset_trainer() → train_proj_adam()`.

- `ps_run.py`  
  Entry point for the Poisson task. Its responsibility is similar to `ac_run.py`, but it targets the 2D Poisson equation and additionally attaches `TimePlotCallback`.

- `approximation_run.py`  
  Entry point for pure function approximation experiments without PDE constraints. It also supports rank analysis, loss/error visualization, and the projection-based training stage.

- `Readme.md`  
  A directory-level documentation file summarizing that this experiment folder focuses on solving PDEs with PINNs / MoE, with particular attention to expert weight dynamics and numerical rank monitoring.

## Equation/

The `Equation/` layer is responsible for **defining the physical problem itself**, including loss construction, sampled data organization, exact solutions, error plotting, and ground-truth / prediction visualization. Its internal files include `__init__.py`, `_base.py`, `allen_cahn.py`, `approximation.py`, `cos.py`, `factory.py`, and `poisson.py`.

- `_base.py`  
  Defines the abstract `BaseEquation` interface, requiring subclasses to implement `compute_loss()`, `get_data()`, `exact_solution()`, `plot_error()`, `plot_ground_truth()`, and `plot_u()`. This gives the layer a stable polymorphic interface.

- `factory.py`  
  Uses `get_equation(args)` to map the command-line argument `--eq` to a concrete equation class. It currently explicitly supports `poisson`, `ac`, and `approximation`.  [__add new equation should change mapping__]

- `poisson.py`  
  **$-∆u=f$**. Implements the 2D Poisson equation, including the forcing term `f(x)`, boundary condition `g(x)`, Laplacian computation, PDE + boundary losses, training data generation, as well as error heatmaps, ground-truth plots, prediction plots, and gate visualization.  

- `allen_cahn.py`  
  **$u_t-\epsilon^2 u_{xx}+u^3-u=s(x,t)$**Implements the 1D Allen–Cahn space-time problem. It uses a manufactured solution to construct the source term, and the loss consists of three parts: PDE residual, periodic boundary conditions, and initial conditions. The data dictionary includes `X_f`, `X_bL`, `X_bR`, `X_i`, `s_f`, and `u_i`.

- `approximation.py`  
  **$f-f^*$** Implements a baseline function-approximation task without PDE differential terms. It essentially minimizes the mean squared error of `u(x) - f(x)` and is used to compare network expressivity and rank behavior.

## Pipeline/

`Pipeline/` currently contains `__init__.py` and `pipeline.py`, serving as the assembly layer of the whole project. It uses lazy loading to manage four components: `model`, `trainer`, `data_loader`, and `data`, and links together `args`, `equation`, and `callbacks` through dependency injection. It also already supports `reset_model()` and `reset_trainer()`, making it convenient to switch training stages or rebuild the model within the same script.

## data/

`data/` currently contains `__init__.py` and `data_loader.py`, and is responsible for low-level sampling utilities. At the current stage, it provides methods such as random sampling inside a 2D box, random sampling on 2D boundaries, and regular-grid interior sampling in 2D, which can be freely combined by different equation classes inside `get_data()`.

## model/

The `model/` layer is responsible for network definitions and feature interfaces. It currently contains `__init__.py`, `factory.py`, `mlp.py`, `moe_d.py`, and `moe_d_w.py`. The factory functions jointly manage both **model instantiation** and the selection of the **penultimate feature getter**, which allows the rank-analysis module to reuse a unified interface across different network architectures.

- `factory.py`  
  `get_model(args)` currently supports `mlp`, `moe_d`, and `moe_d_w`, while `get_feature_getter(args)` returns the corresponding feature extraction function for later Gram / epsilon-rank computation.

- `mlp.py`  
  Implements a standard multilayer perceptron. Structurally, it is divided into a feature extraction body and a bias-free output head, and provides interfaces such as `forward_penultimate()` / `mlp_penultimate_getter` for extracting penultimate-layer representations.

- `moe_d.py`  
  Implements a dense-gate MoE, organized into four levels: `Expert`, `Gating`, `MoE`, and `MOE_dense`. It also provides `moe_penultimate_getter()`, whose features are formed by concatenating **each expert’s penultimate hidden activation × gating weight**.

- `moe_d_w.py`  
  Implements another dense-weight MoE variant, preserving the mixed expert + gating structure, and serves as the second MoE candidate model in the current experiments.

## train/

`train/` currently contains `LM.py`, `__init__.py`, `proj.py`, and `trainer.py`. This layer is responsible for training loops and projection-related algorithms. `trainer.py` is the core training orchestrator, uniformly managing optimizers, schedulers, callback triggering across different phases, and extracting `feature_getter` from `RankCallback` to support projection-based training. `proj.py` implements the minimum eigenpair, `j_min`-related functions based on feature mappings, and related utilities. `LM.py` is still a placeholder at this stage.

## metrics/

`metrics/` currently contains `__init__.py` and `epsilon_rank2D.py`. Its core functionality is to construct the Gram matrix based on 2D trapezoidal integration and automatically compute the epsilon-rank of model features. It already implements 2D grid generation, 2D trapezoidal weights, and `epsilon_rank_model_2d_trapz_auto()`. This module is exactly the core metric component for the project’s current focus on **numerical rank monitoring / expert collapse diagnosis**.

## viz/

`viz/` currently contains `__init__.py`, `callbacks.py`, `checkpoint_callback.py`, `error_plot_callback.py`, `loss_plot_callback.py`, `rank_callback.py`, and `time_plot_callback.py`, forming a fairly complete callback system.

- `callbacks.py`  
  Defines the base callback interface, including `on_train_begin`, `on_iter_end`, `on_train_end`, and `on_phase_begin`.

- `checkpoint_callback.py`  
  Saves checkpoints at phase-dependent frequencies and maintains `_log_model/last.pt`.

- `error_plot_callback.py`  
  Calls the visualization interfaces in the equation layer before, during, and after training to generate ground-truth plots, error plots, and prediction plots. If the model is an MoE, it additionally plots the gate distribution.

- `loss_plot_callback.py`  
  Records the history of multiple loss terms and generates loss curves in logarithmic scale.

- `rank_callback.py`  
  Periodically calls `epsilon_rank_model_2d_trapz_auto()` to compute rank and feature spectra, producing `rank_curve.png` and `rank_distribution.png`. It is one of the most distinctive analysis modules in the current experiment directory.

- `time_plot_callback.py`  
  Plots loss curves against time rather than iteration count, making it useful for comparing the time efficiency of different training stages and methods.

## outputs/

`outputs/` is the experiment results directory. According to the current README, its output structure is rooted at `save_dir/`, and includes at least the checkpoint files under `_log_model/`, together with the logs and figure outputs generated during training.

---

If needed, I can next compress this into a more formal CHANGELOG style, such as using sections like `### Added / ### Refactored / ### Current layout`.