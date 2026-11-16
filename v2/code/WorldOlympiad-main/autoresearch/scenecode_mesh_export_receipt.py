"""Record the actual Blender mesh exporter settings without a Blender dependency."""
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote


def export_geometry_hashes(path):
    path = Path(path).resolve()
    files = {path.name: path}
    if path.suffix.lower() == ".gltf":
        for buffer in json.loads(path.read_text()).get("buffers", []):
            uri = buffer.get("uri", "")
            if uri.startswith("data:"):
                continue
            if not uri or "://" in uri:
                raise ValueError("export frame receipts require local glTF buffers")
            files[uri] = path.parent / unquote(uri)
    return {name: hashlib.sha256(file.read_bytes()).hexdigest() for name, file in files.items()}


def record_blender_y_up_export(path):
    """Called only after evaluated-mesh export with explicit Y-up settings."""
    path = Path(path).resolve()
    if path.suffix.lower() not in {".obj", ".gltf"}:
        raise ValueError("unsupported paired part export format")
    receipt = {"schema_version": "scenecode_blender_mesh_export.v1",
               "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "geometry_file_sha256": export_geometry_hashes(path),
               "applied_modifiers": True,
               "blender_to_mesh_basis": [[1, 0, 0], [0, 0, 1], [0, -1, 0]]}
    destination = path.with_suffix(path.suffix + ".frame.json")
    destination.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return str(destination)
