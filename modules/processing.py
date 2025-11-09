"""Compatibility shim for legacy imports.

This module re-exports the processing helpers that now live in
:mod:`nodes`. Downstream integrations should import directly from
``nodes`` instead of ``modules.processing``.
"""

from __future__ import annotations

import warnings

from nodes import (
    MODES,
    SEAM_FIX_MODES,
    Processed,
    StableDiffusionProcessing,
    USDUMode,
    USDUSFMode,
    fix_seed,
    process_images,
    sample,
)

warnings.warn(
    "modules.processing is deprecated; import from nodes instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "MODES",
    "SEAM_FIX_MODES",
    "Processed",
    "StableDiffusionProcessing",
    "USDUMode",
    "USDUSFMode",
    "fix_seed",
    "process_images",
    "sample",
]
