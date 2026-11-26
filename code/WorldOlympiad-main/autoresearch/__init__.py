"""SceneCode-to-PyBullet autoresearch data pipeline."""

__all__ = ["EpisodeManifest", "EpisodeSpec", "PhysicsReport", "WorldIR"]


def __getattr__(name):
    # Pure file/geometry receipts also run in Blender's bundled Python. Loading
    # those helpers must not require the unrelated Pydantic task models.
    if name in __all__:
        from . import models
        value = getattr(models, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
