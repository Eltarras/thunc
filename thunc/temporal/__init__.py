"""Optional Temporal execution. Install ``thunc[temporal]`` to use the runtime.

Imports are lazy so the core and Workflow sandbox never load worker dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import Handle as Handle
    from .client import Runtime as Runtime
    from .models import DurableError as DurableError
    from .models import DurableRun as DurableRun
    from .registry import Registry as Registry
    from .worker import Worker as Worker


def __getattr__(name: str) -> Any:
    import importlib

    modules = {
        "Runtime": "client",
        "Handle": "client",
        "Registry": "registry",
        "Worker": "worker",
        "DurableRun": "models",
        "DurableError": "models",
    }
    if name not in modules:
        raise AttributeError(name)
    try:
        return getattr(importlib.import_module(f"{__name__}.{modules[name]}"), name)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("temporalio"):
            raise ImportError("Install the Temporal extra: pip install 'thunc[temporal]'") from exc
        raise
