"""Lazy domain selection shared by inference, reward scoring and native rollouts."""
from importlib import import_module
from types import ModuleType


DOMAIN_MODULES = {
    "math": ".retool",
    "searchqa": ".searchqa_domain",
    "chess": ".chess_domain",
    "alfworld": ".alfworld_domain",
}


def get_domain(name: str = "math") -> ModuleType:
    if name not in DOMAIN_MODULES:
        raise ValueError(f"Unsupported domain: {name}")
    return import_module(DOMAIN_MODULES[name], package=__package__)
