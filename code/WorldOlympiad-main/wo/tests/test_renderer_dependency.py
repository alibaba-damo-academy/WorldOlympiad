from types import SimpleNamespace

import pytest

from wo.eval.renderer_dependency import (
    bind_gsplat_rasterization,
    load_gsplat_rasterization,
    require_gsplat_rasterization,
)


def test_renderer_dependency_returns_real_callable_rasterizer():
    rasterize = lambda **kwargs: kwargs
    assert require_gsplat_rasterization(SimpleNamespace(rasterization=rasterize)) is rasterize


def test_missing_gsplat_fails_explicitly_before_geometry_scoring():
    with pytest.raises(ImportError, match="requires callable gsplat.rasterization"):
        require_gsplat_rasterization(SimpleNamespace())


def test_non_callable_rasterization_is_not_treated_as_a_measurement():
    with pytest.raises(ImportError, match="install the pinned gsplat CUDA build"):
        require_gsplat_rasterization(SimpleNamespace(rasterization=None))


def test_renderer_loader_imports_the_real_module_contract():
    rasterize = lambda **kwargs: kwargs
    imported = []

    def fake_import(name):
        imported.append(name)
        return SimpleNamespace(rasterization=rasterize)

    assert load_gsplat_rasterization(fake_import) is rasterize
    assert imported == ["gsplat"]


def test_missing_gsplat_import_fails_before_renderer_call():
    def missing_import(name):
        raise ModuleNotFoundError("gsplat unavailable")

    with pytest.raises(ImportError, match="requires gsplat.rasterization") as exc:
        load_gsplat_rasterization(missing_import)
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)


def test_backend_binds_validated_rasterizer_into_da3_module():
    renderer = SimpleNamespace()
    rasterize = lambda **kwargs: kwargs
    assert bind_gsplat_rasterization(renderer, rasterize) is rasterize
    assert renderer.rasterization is rasterize


def test_backend_refuses_to_bind_missing_rasterizer():
    with pytest.raises(ImportError, match="requires callable gsplat.rasterization"):
        bind_gsplat_rasterization(SimpleNamespace(), None)
