#\Equation/factory.py
from .poisson import PoissonEquation
from .allen_cahn import AllenCahnEquation
from .approximation import FunctionFitEquation
from .heat_ms2d import Heat2DMSEquation
from .dcei import DCEIEquation
from .poisson10d import POSSION10dEquation
from .helmholtz import HelmholtzEquation
from .NS import KovasznayEquation
from .burger1d import Burgers1DEquation
def get_equation(args):
    """
    根据命令行参数 --eq 自动选择对应的方程类
    """
    mapping = {
        "poisson": PoissonEquation,
         "ac" : AllenCahnEquation,
         "approximation":FunctionFitEquation,
         "heat_ms2d": Heat2DMSEquation,
         "dcei": DCEIEquation,
         "poisson10d": POSSION10dEquation,
         "helmholtz": HelmholtzEquation,
         "navier_stokes":KovasznayEquation,
         "burgers1d": Burgers1DEquation
    }
    target_class = mapping.get(args.eq.lower())
    if not target_class:
        raise ValueError(f"Equation {args.eq} is not defined in factory.")
    return target_class(args)