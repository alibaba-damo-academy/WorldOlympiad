import base64
from io import BytesIO
import pytest
from PIL import Image
from wo.agent import tools as T


def test_view_image_returns_pixels_as_multimodal_content(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    p = tmp_path / "actual_native_frame.png"
    Image.new("RGB", (12, 8), (23, 140, 81)).save(p)
    result = T.view_image(str(p))
    assert T.view_image in T.ALL
    assert result[0]["type"] == "text" and str(p) in result[0]["text"]
    assert result[1]["type"] == "image"
    prefix, encoded = result[1]["image_url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (12, 8) and image.getpixel((2, 3)) == (23, 140, 81)


def test_view_image_respects_workspace_and_secret_symlinks(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(T, "WORKSPACE", root)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    with pytest.raises(ValueError, match="inside"):
        T.view_image(str(outside))
    secret = root / ".secrets"
    secret.mkdir()
    hidden = secret / "secret.png"
    Image.new("RGB", (2, 2)).save(hidden)
    link = root / "visible.png"
    link.symlink_to(hidden)
    with pytest.raises(ValueError, match="secrets"):
        T.view_image(str(link))


def test_view_image_rejects_invalid_or_oversized_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    invalid = tmp_path / "not_an_image.png"
    invalid.write_text("not pixels")
    with pytest.raises(OSError):
        T.view_image(str(invalid))
    big = tmp_path / "large.png"
    with big.open("wb") as f:
        f.truncate(8 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="8 MiB"):
        T.view_image(str(big))
