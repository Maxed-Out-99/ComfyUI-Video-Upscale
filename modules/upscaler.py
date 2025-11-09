"""Compatibility shim for legacy imports.

Upscaler helpers now live in :mod:`nodes`. Import from there directly in
new code.
"""

from __future__ import annotations

import warnings

from nodes import Upscaler, UpscalerData

warnings.warn(
    "modules.upscaler is deprecated; import from nodes instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Upscaler", "UpscalerData"]
