"""Harness-R1 adaptation for the frozen SearchQA target."""

from .patch import (
    HOOK_NAMES,
    PatchValidationError,
    base_patch,
    compile_patch,
    extract_and_normalize_patch,
    merge_patches,
    normalize_patch,
)

__all__ = [
    "HOOK_NAMES",
    "PatchValidationError",
    "base_patch",
    "compile_patch",
    "extract_and_normalize_patch",
    "merge_patches",
    "normalize_patch",
]
