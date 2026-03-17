

# Table of Contents
- [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Running codes](#running-codes)
  - [Equation](#equation)
    - [Equation Methods](#equation-methods)
  - [Pipeline](#pipeline)
  - [data/](#data)
  - [model/](#model)
  - [train](#train)
    - [🚀 Optimization methods](#-optimization-methods)
      - [___Levenber-Marquardt method___](#levenber-marquardt-method)
      - [___Adam___](#adam)
      - [___projection Adam___](#projection-adam)
  - [metrics](#metrics)
  - [viz](#viz)
  - [outputs](#outputs)
  - [Version update](#version-update)
    - [\[__0.01__\] - 2026.3.8](#001---202638)
    - [\[__0.02__\] - 2026.3.17](#002---2026317)

## Overview

`dlpdes/` has now evolved into a relatively complete PINN experimental framework. Its overall structure is organized by layers: **equation definition → data generation → model construction → training orchestration → metric analysis → visualization callbacks**. Each script first parses arguments, constructs the equation object and feature extractor through factory functions, then assembles the `Pipeline` together with multiple callbacks.

## Running codes

- `ac_run.py`  
  Entry point for the Allen–Cahn task. It defines spatio-temporal sampling, PDE/BC/IC weights, model hyperparameters, training frequencies, and rank-analysis settings, while wiring together `ErrorPlotCallback`, `LossPlotCallback`, `CheckpointCallback`, and `RankCallback`. The training flow is `train_adam() → reset_model() → reset_trainer() → train_proj_adam()`.

- `ps_run.py`  
  Entry point for the Poisson task. Its responsibility is similar to `ac_run.py`, but it targets the 2D Poisson equation and additionally attaches `TimePlotCallback`.

- `approximation_run.py`  
  Entry point for pure function approximation experiments without PDE constraints. It also supports rank analysis, loss/error visualization, and the projection-based training stage.

- `Readme.md`  
  A directory-level documentation file summarizing that this experiment folder focuses on solving PDEs with PINNs / MoE, with particular attention to expert weight dynamics and numerical rank monitoring.


## Equation

The `Equation/` layer is responsible for **defining the physical problem itself**, including:

- Loss construction

---

### Equation Methods

-  `compute_loss(model, data: dict)`

    This function computes losses based on the given input data and returns a dictionary with the following structure:

    ```python
    {
      "losses": {
        "total": <total loss with gradient>, # ready to backpropagation
        "pde" (optional): <PDE loss component>,
        "bc"  (optional): <Boundary loss component>
      },
      "residuals": {
        "all": <residual values>  # ready for lm methods
      }
    }
    ```

## Pipeline

`Pipeline/` 

## data/

`data/` 

## model/

`model/` 

## train

`train/` 
### 🚀 Optimization methods  
#### ___Levenber-Marquardt method___
  a quasi-newton method $h \leftarrow (J^T J + \lambda I )^{-1} J^T r$ , phase set `lm`.
  ```python
  Pipeline.trainer.train_lm(Pipeline.data) # apply train_lm to train the model in trainer
  ```
  **Parameter:**  

  | param name | default | type | description |
  | :--- | :--- | :--- | :--- |
  | `lm_epochs` | `1500` | `int` | **Number of iterations:** total number of LM training iterations. |
  | `lm_gama` | `2.0` | `float` | **Damping update coefficient ($\gamma$):** must satisfy $\gamma > 1$. |
  | `lm_yita1` | `1e-16` | `float` | **Acceptance threshold ($\eta_1$):** lower bound for the update quality, i.e. require $\rho > \eta_1$. |
  | `lm_yita2` | `1e-6` | `float` | **Gradient threshold ($\eta_2$):** threshold used in the gradient update condition. |
  | `lm_min_miu` | `1e-16` | `float` | **Minimum $\mu$:** lower bound of the damping parameter. |
  | `lm_max_miu` | `1e100` | `float` | **Maximum $\mu$:** upper bound of the damping parameter. |
  | `lm_train_tol` | `1e-5` | `float` | **Stopping criterion:** stop training when the gradient norm is below this tolerance. <span style="color: yellow;">Not enabled yet.</span> |
  | `lm_miu` | `1e-3` | `float` | **Damping parameter ($\mu$):** increased when an update is rejected and decreased when an update is accepted. |
  | `lm_beta_train` | `False` | `bool` | **Subset update:** set to `True` to enable subset-based LM training. |

  The details of Levenberg-Marquardt method can be found in lm notes.

**💡 Note :** 
  - The loss and residuals should be computed using `torch.func.jacrev`. In the equation-level implementation of `compute_loss`, include a mode such as `jacrev`.
  - The residuals are stored in `loss_dict["residuals"]["all"]`.
  - To ensure good training performance, the code sets the default value of $\beta$ to $\frac{1}{2} N_{\text{params}}$ in this regime.
  
---
#### ___Adam___

  adam optimizer, phase set `adam`
  ```python
      Pipeline.trainer.train_adam(Pipeline.data)  # train the model using adam 
  ```
  The method includes function: `_step_adam()` and `train_adam` 
  | param name | default | type | description |
  | :--- | :--- | :--- | :--- |
  | `adam_iters` | `10000` | `int` | **Number of iterations:**: total number of Adam training iterations. |
  | `adam_lr` | `1e-3` | `float` | **learning rate**: learning rate of adam |
  | `use_scheduler` | `True` | `bool` | **Use scheduler**: reduce the lr in during the training |
  | `sc_step_size` | `5000` | `int` | **scheduler frequency**: reduce the lr freq |
  | `sc_gamma` | `0.7` | `float` | **scheduler gamma**: the cofficients of reduce lr |


  **💡 Note :** 
  - adam implemented by `torch.autograd.grad` , `compute_loss`'s mode set in `backward`





      

---
#### ___projection Adam___ 
  <span style="color: yellow;">Not enabled yet</span>


  


## metrics

`metrics/` currently contains `__init__.py` and `epsilon_rank2D.py`. Its core functionality is to construct the Gram matrix based on 2D trapezoidal integration and automatically compute the epsilon-rank of model features. It already implements 2D grid generation, 2D trapezoidal weights, and `epsilon_rank_model_2d_trapz_auto()`. This module is exactly the core metric component for the project’s current focus on **numerical rank monitoring / expert collapse diagnosis**.

## viz

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

## outputs

`outputs/` is the experiment results directory. Its output structure is rooted at `save_dir/`, and includes at least the checkpoint files under `_log_model/`, together with the logs and figure outputs generated during training.



## Version update

### [__0.01__] - 2026.3.8  
  - First update

### [__0.02__] - 2026.3.17

  - Updated the loss dictionary structure.
  - Implemented the LM optimizer.
  - Modified `compute_loss` so that it builds the gradient graph according to the selected mode.
  - Added new files: `train/utils.py` and `train/lm.py`.



