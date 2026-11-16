"""Preflight helpers for optional 3D rendering dependencies."""

from __future__ import annotations

import importlib
from typing import Any, Callable


def require_gsplat_rasterization(renderer: Any) -> Any:
    """Return gsplat's rasterizer or fail before geometry scoring can degrade.

    DA3's renderer historically logged an ImportError but continued module import,
    leaving an unbound global that later raised ``NameError`` inside a scoring
    fallback. Check the actual imported renderer module at the scoring boundary so
    missing gsplat is an explicit infrastructure error, never a zero measurement.
    """
    rasterization = getattr(renderer, "rasterization", None)
    if not callable(rasterization):
        raise ImportError(
            "3D geometry scoring requires callable gsplat.rasterization; "
            "install the pinned gsplat CUDA build in the worldeval environment"
        )
    return rasterization


def load_gsplat_rasterization(import_module: Callable[[str], Any] | None = None) -> Any:
    """Import and validate gsplat's production rasterizer without a latent NameError.

    The vendored DA3 renderer used to catch ImportError and continue module import,
    leaving a global that failed much later during geometry scoring. Fail at the
    import boundary, preserving the original dependency exception as the cause.
    ``import_module`` is injectable solely to test this boundary without CUDA.
    """
    if import_module is None:
        import_module = importlib.import_module
    try:
        gsplat = import_module("gsplat")
    except ImportError as exc:
        raise ImportError(
            "3D geometry rendering requires gsplat.rasterization; install the "
            "pinned gsplat CUDA build in the worldeval environment"
        ) from exc
    return require_gsplat_rasterization(gsplat)


def bind_gsplat_rasterization(renderer: Any, rasterization: Any) -> Any:
    """Bind a validated gsplat callable into DA3's renderer module namespace."""
    if not callable(rasterization):
        raise ImportError(
            "3D geometry rendering requires callable gsplat.rasterization; "
            "install the pinned gsplat CUDA build in the worldeval environment"
        )
    renderer.rasterization = rasterization
    return rasterization
