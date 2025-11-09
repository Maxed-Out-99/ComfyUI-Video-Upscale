"""Compatibility proxy for the shared namespace used by USDU helpers."""

from __future__ import annotations

import sys
from types import ModuleType

from nodes import Options, State, shared as _shared


class _SharedModule(ModuleType):
    """Module proxy that forwards attribute access to :mod:`nodes`."""

    Options = Options
    State = State

    def __getattr__(self, name):  # pragma: no cover - thin forwarding layer
        if hasattr(_shared, name):
            return getattr(_shared, name)
        raise AttributeError(name) from None

    def __setattr__(self, name, value):  # pragma: no cover - thin forwarding layer
        if hasattr(_shared, name):
            setattr(_shared, name, value)
        else:
            super().__setattr__(name, value)


_module = _SharedModule(__name__)
_module.__dict__["__file__"] = __file__
_module.__dict__["__package__"] = __package__
_module.__dict__["__spec__"] = None

# Seed default attributes so direct reads behave as expected.
_module.opts = _shared.opts
_module.state = _shared.state
_module.sd_upscalers = _shared.sd_upscalers
_module.actual_upscaler = _shared.actual_upscaler
_module.batch = _shared.batch
_module.batch_as_tensor = _shared.batch_as_tensor

sys.modules[__name__] = _module
