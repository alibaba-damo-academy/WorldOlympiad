"""GLTF generation utilities for textured walls and floors."""

import base64
import logging
import os
import warnings

from pathlib import Path
from typing import Literal

import numpy as np

from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    GLTF2,
    UNSIGNED_INT,
    Accessor,
    Attributes,
    Buffer,
    BufferView,
    Image,
    Material as GltfMaterial,
    Mesh,
    Node,
    NormalMaterialTexture,
    PbrMetallicRoughness,
    Primitive,
    Scene,
    Texture,
    TextureInfo,
)

from scenecode.utils.material import Material

console_logger = logging.getLogger(__name__)


def get_zup_to_yup_matrix() -> np.ndarray:
    """Get the 4x4 transformation matrix for Z-up to Y-up coordinate conversion.

    Returns:
        4x4 numpy array representing the Z-up to Y-up transformation.
        Transformation: (x, y, z)_Zup → (x, z, -y)_Yup
    """
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
    )


def zup_to_yup_transform(vertices_zup: np.ndarray) -> np.ndarray:
    """
    Transform vertices from Drake Z-up coordinates to GLTF Y-up coordinates.

    Drake uses Z-up coordinate system, while GLTF standard uses Y-up.
    Drake automatically converts Y-up GLTF to Z-up when loading meshes.
    This function applies the inverse transform so that GLTF files are
    in standard Y-up format.

    Transformation: (x, y, z)_Zup → (x, z, -y)_Yup

    Args:
        vertices_zup: Nx3 array of vertices in Drake Z-up coordinates.

    Returns:
        Nx3 array of vertices in GLTF Y-up coordinates.
    """
    vertices_yup = np.column_stack(
        [
            vertices_zup[:, 0],  # X unchanged
            vertices_zup[:, 2],  # Z becomes Y
            -vertices_zup[:, 1],  # -Y becomes Z
        ]
    )
    return vertices_yup.astype(np.float32)


def create_gltf_from_mesh_data(
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
    color_uri: str,
    normal_uri: str,
    roughness_uri: str,
    output_path: Path,
) -> None:
    """Create GLTF file from mesh data with PBR material.

    Args:
        vertices: Nx3 vertex positions in Y-up coordinates.
        normals: Nx3 vertex normals in Y-up coordinates.
        uvs: Nx2 texture coordinates.
        indices: Triangle indices.
        color_uri: Relative path to color texture.
        normal_uri: Relative path to normal texture.
        roughness_uri: Relative path to roughness texture.
        output_path: Where to save the GLTF file.
    """
    vertices_binary = vertices.astype(np.float32).tobytes()
    normals_binary = normals.astype(np.float32).tobytes()
    uvs_binary = uvs.astype(np.float32).tobytes()
    indices_binary = indices.astype(np.uint32).tobytes()

    buffer_data = vertices_binary + normals_binary + uvs_binary + indices_binary
    buffer_length = len(buffer_data)

    vertices_buffer_view = BufferView(
        buffer=0,
        byteOffset=0,
        byteLength=len(vertices_binary),
        target=ARRAY_BUFFER,
    )

    normals_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertices_binary),
        byteLength=len(normals_binary),
        target=ARRAY_BUFFER,
    )

    uvs_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertices_binary) + len(normals_binary),
        byteLength=len(uvs_binary),
        target=ARRAY_BUFFER,
    )

    indices_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertices_binary) + len(normals_binary) + len(uvs_binary),
        byteLength=len(indices_binary),
        target=ELEMENT_ARRAY_BUFFER,
    )

    vertices_accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=FLOAT,
        count=len(vertices),
        type="VEC3",
        min=vertices.min(axis=0).tolist(),
        max=vertices.max(axis=0).tolist(),
    )

    normals_accessor = Accessor(
        bufferView=1,
        byteOffset=0,
        componentType=FLOAT,
        count=len(normals),
        type="VEC3",
    )

    uvs_accessor = Accessor(
        bufferView=2,
        byteOffset=0,
        componentType=FLOAT,
        count=len(uvs),
        type="VEC2",
    )

    indices_accessor = Accessor(
        bufferView=3,
        byteOffset=0,
        componentType=UNSIGNED_INT,
        count=len(indices),
        type="SCALAR",
    )

    material = GltfMaterial(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=0),
            metallicRoughnessTexture=TextureInfo(index=2),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
        normalTexture=NormalMaterialTexture(index=1),
        doubleSided=False,
    )

    gltf_textures = [
        Texture(source=0),
        Texture(source=1),
        Texture(source=2),
    ]

    images = [
        Image(uri=color_uri),
        Image(uri=normal_uri),
        Image(uri=roughness_uri),
    ]

    primitive = Primitive(
        attributes=Attributes(POSITION=0, NORMAL=1, TEXCOORD_0=2),
        indices=3,
        material=0,
    )

    mesh = Mesh(primitives=[primitive])

    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[mesh],
        materials=[material],
        textures=gltf_textures,
        images=images,
        accessors=[
            vertices_accessor,
            normals_accessor,
            uvs_accessor,
            indices_accessor,
        ],
        bufferViews=[
            vertices_buffer_view,
            normals_buffer_view,
            uvs_buffer_view,
            indices_buffer_view,
        ],
        buffers=[Buffer(byteLength=buffer_length)],
    )

    gltf.set_binary_blob(buffer_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*contains a binary blob.*", category=UserWarning
        )
        gltf.save(str(output_path))


def create_glb_from_mesh_data(
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
    color_texture_path: Path,
    normal_texture_path: Path,
    roughness_texture_path: Path,
    output_path: Path,
) -> None:
    """Create self-contained GLB file from mesh data with embedded PBR textures.

    This creates a binary GLTF (GLB) file with all textures embedded as base64
    data URIs. The resulting file is completely self-contained with no external
    dependencies, making it suitable for validation rendering via BlenderServer.

    Args:
        vertices: Nx3 vertex positions in Y-up coordinates.
        normals: Nx3 vertex normals in Y-up coordinates.
        uvs: Nx2 texture coordinates.
        indices: Triangle indices.
        color_texture_path: Path to color/albedo texture (JPG/PNG).
        normal_texture_path: Path to normal map texture (JPG/PNG).
        roughness_texture_path: Path to roughness texture (JPG/PNG).
        output_path: Where to save the GLB file. Must have .glb extension.

    Raises:
        ValueError: If output_path doesn't have .glb extension.
        FileNotFoundError: If any texture file doesn't exist.
    """
    if output_path.suffix.lower() != ".glb":
        raise ValueError(f"Output path must have .glb extension: {output_path}")

    # Verify texture files exist.
    for tex_path in [color_texture_path, normal_texture_path, roughness_texture_path]:
        if not tex_path.exists():
            raise FileNotFoundError(f"Texture file not found: {tex_path}")

    # Read and encode textures as base64 data URIs.
    def encode_texture(path: Path) -> str:
        """Read texture file and encode as base64 data URI."""
        with open(path, "rb") as f:
            data = f.read()
        # Determine MIME type from extension.
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        else:
            mime_type = "application/octet-stream"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    color_data_uri = encode_texture(color_texture_path)
    normal_data_uri = encode_texture(normal_texture_path)
    roughness_data_uri = encode_texture(roughness_texture_path)

    # Build geometry buffer.
    vertices_binary = vertices.astype(np.float32).tobytes()
    normals_binary = normals.astype(np.float32).tobytes()
    uvs_binary = uvs.astype(np.float32).tobytes()
    indices_binary = indices.astype(np.uint32).tobytes()

    buffer_data = vertices_binary + normals_binary + uvs_binary + indices_binary
    buffer_length = len(buffer_data)

    vertices_buffer_view = BufferView(
        buffer=0,
        byteOffset=0,
        byteLength=len(vertices_binary),
        target=ARRAY_BUFFER,
    )

    normals_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertices_binary),
        byteLength=len(normals_binary),
        target=ARRAY_BUFFER,
    )

    uvs_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertices_binary) + len(normals_binary),
        byteLength=len(uvs_binary),
        target=ARRAY_BUFFER,
    )

    indices_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertices_binary) + len(normals_binary) + len(uvs_binary),
        byteLength=len(indices_binary),
        target=ELEMENT_ARRAY_BUFFER,
    )

    vertices_accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=FLOAT,
        count=len(vertices),
        type="VEC3",
        min=vertices.min(axis=0).tolist(),
        max=vertices.max(axis=0).tolist(),
    )

    normals_accessor = Accessor(
        bufferView=1,
        byteOffset=0,
        componentType=FLOAT,
        count=len(normals),
        type="VEC3",
    )

    uvs_accessor = Accessor(
        bufferView=2,
        byteOffset=0,
        componentType=FLOAT,
        count=len(uvs),
        type="VEC2",
    )

    indices_accessor = Accessor(
        bufferView=3,
        byteOffset=0,
        componentType=UNSIGNED_INT,
        count=len(indices),
        type="SCALAR",
    )

    material = GltfMaterial(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=0),
            metallicRoughnessTexture=TextureInfo(index=2),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
        normalTexture=NormalMaterialTexture(index=1),
        doubleSided=False,
    )

    gltf_textures = [
        Texture(source=0),
        Texture(source=1),
        Texture(source=2),
    ]

    # Use data URIs for embedded textures.
    images = [
        Image(uri=color_data_uri),
        Image(uri=normal_data_uri),
        Image(uri=roughness_data_uri),
    ]

    primitive = Primitive(
        attributes=Attributes(POSITION=0, NORMAL=1, TEXCOORD_0=2),
        indices=3,
        material=0,
    )

    mesh = Mesh(primitives=[primitive])

    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[mesh],
        materials=[material],
        textures=gltf_textures,
        images=images,
        accessors=[
            vertices_accessor,
            normals_accessor,
            uvs_accessor,
            indices_accessor,
        ],
        bufferViews=[
            vertices_buffer_view,
            normals_buffer_view,
            uvs_buffer_view,
            indices_buffer_view,
        ],
        buffers=[Buffer(byteLength=buffer_length)],
    )

    gltf.set_binary_blob(buffer_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as GLB (binary GLTF). pygltflib automatically detects .glb extension.
    gltf.save(str(output_path))


def _create_box_mesh_gltf(
    width: float,
    depth: float,
    height: float,
    material: Material,
    output_path: Path,
    texture_scale: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> None:
    """
    Create a 3D box mesh with PBR material and save as GLTF.

    Generates a box with 6 textured faces in Drake Z-up coordinates at the
    specified world position, then transforms to Y-up GLTF standard before export.

    Args:
        width: X dimension in meters (Drake Z-up).
        depth: Y dimension in meters (Drake Z-up).
        height: Z dimension in meters (Drake Z-up).
        material: PBR material with textures.
        output_path: Where to save the GLTF file.
        texture_scale: Meters per texture tile (for UV tiling).
        center_x: X position of box center in Drake Z-up coordinates.
        center_y: Y position of box center in Drake Z-up coordinates.
        center_z: Z position of box center in Drake Z-up coordinates.
    """
    textures = material.get_all_textures()

    # Convert texture paths to URIs relative to output GLTF.
    output_abs = output_path.resolve()
    color_uri = Path(
        os.path.relpath(textures["color"].resolve(), output_abs.parent)
    ).as_posix()
    normal_uri = Path(
        os.path.relpath(textures["normal"].resolve(), output_abs.parent)
    ).as_posix()
    roughness_uri = Path(
        os.path.relpath(textures["roughness"].resolve(), output_abs.parent)
    ).as_posix()

    # Create box vertices in Drake Z-up coordinates.
    # Use 24 vertices (4 per face) to allow independent UV mapping per face.
    half_w = width / 2.0
    half_d = depth / 2.0
    half_h = height / 2.0

    # Define 24 vertices for box (4 per face, counter-clockwise from outside).
    # Face order: Front (+Y), Back (-Y), Right (+X), Left (-X), Top (+Z), Bottom (-Z).
    vertices_zup = np.array(
        [
            # Front face (+Y): vertices 0-3
            [center_x + half_w, center_y + half_d, center_z - half_h],
            [center_x - half_w, center_y + half_d, center_z - half_h],
            [center_x - half_w, center_y + half_d, center_z + half_h],
            [center_x + half_w, center_y + half_d, center_z + half_h],
            # Back face (-Y): vertices 4-7
            [center_x - half_w, center_y - half_d, center_z - half_h],
            [center_x + half_w, center_y - half_d, center_z - half_h],
            [center_x + half_w, center_y - half_d, center_z + half_h],
            [center_x - half_w, center_y - half_d, center_z + half_h],
            # Right face (+X): vertices 8-11
            [center_x + half_w, center_y - half_d, center_z - half_h],
            [center_x + half_w, center_y + half_d, center_z - half_h],
            [center_x + half_w, center_y + half_d, center_z + half_h],
            [center_x + half_w, center_y - half_d, center_z + half_h],
            # Left face (-X): vertices 12-15
            [center_x - half_w, center_y + half_d, center_z - half_h],
            [center_x - half_w, center_y - half_d, center_z - half_h],
            [center_x - half_w, center_y - half_d, center_z + half_h],
            [center_x - half_w, center_y + half_d, center_z + half_h],
            # Top face (+Z): vertices 16-19
            [center_x - half_w, center_y - half_d, center_z + half_h],
            [center_x + half_w, center_y - half_d, center_z + half_h],
            [center_x + half_w, center_y + half_d, center_z + half_h],
            [center_x - half_w, center_y + half_d, center_z + half_h],
            # Bottom face (-Z): vertices 20-23
            [center_x - half_w, center_y + half_d, center_z - half_h],
            [center_x + half_w, center_y + half_d, center_z - half_h],
            [center_x + half_w, center_y - half_d, center_z - half_h],
            [center_x - half_w, center_y - half_d, center_z - half_h],
        ],
        dtype=np.float32,
    )

    # Transform to Y-up for GLTF.
    vertices = zup_to_yup_transform(vertices_zup)

    # fmt: off
    indices = np.array([
        0, 1, 2, 0, 2, 3,        # Front
        4, 5, 6, 4, 6, 7,        # Back
        8, 9, 10, 8, 10, 11,     # Right
        12, 13, 14, 12, 14, 15,  # Left
        16, 17, 18, 16, 18, 19,  # Top
        20, 21, 22, 20, 22, 23,  # Bottom
    ], dtype=np.uint32)
    # fmt: on

    # Calculate normals for each face in Drake Z-up.
    # Each face's 4 vertices share the same outward-pointing normal.
    normals_zup_per_face = np.array(
        [
            [0, 1, 0],  # Front (+Y)
            [0, -1, 0],  # Back (-Y)
            [1, 0, 0],  # Right (+X)
            [-1, 0, 0],  # Left (-X)
            [0, 0, 1],  # Top (+Z)
            [0, 0, -1],  # Bottom (-Z)
        ],
        dtype=np.float32,
    )

    # Replicate each normal 4 times (one per vertex in the face).
    VERTICES_PER_FACE = 4
    normals_zup = np.repeat(normals_zup_per_face, VERTICES_PER_FACE, axis=0)

    # Transform normals to Y-up.
    normals = zup_to_yup_transform(normals_zup)

    # UV coordinates: each face uses its physical dimensions for tiling.
    uv_front_back = (width / texture_scale, height / texture_scale)
    uv_left_right = (depth / texture_scale, height / texture_scale)
    uv_top_bottom = (width / texture_scale, depth / texture_scale)

    # Each face maps UVs from (0,0) to its dimension ratio.
    # Pattern for each face: bottom-left, bottom-right, top-right, top-left.
    uvs = np.array(
        [
            # Front/back faces: width × height
            [0, 0],
            [uv_front_back[0], 0],
            [uv_front_back[0], uv_front_back[1]],
            [0, uv_front_back[1]],
            [0, 0],
            [uv_front_back[0], 0],
            [uv_front_back[0], uv_front_back[1]],
            [0, uv_front_back[1]],
            # Left/right faces: depth × height
            [0, 0],
            [uv_left_right[0], 0],
            [uv_left_right[0], uv_left_right[1]],
            [0, uv_left_right[1]],
            [0, 0],
            [uv_left_right[0], 0],
            [uv_left_right[0], uv_left_right[1]],
            [0, uv_left_right[1]],
            # Top/bottom faces: width × depth
            [0, 0],
            [uv_top_bottom[0], 0],
            [uv_top_bottom[0], uv_top_bottom[1]],
            [0, uv_top_bottom[1]],
            [0, 0],
            [uv_top_bottom[0], 0],
            [uv_top_bottom[0], uv_top_bottom[1]],
            [0, uv_top_bottom[1]],
        ],
        dtype=np.float32,
    )

    # Pack all data into binary buffer.
    vertices_binary = vertices.tobytes()
    normals_binary = normals.tobytes()
    uvs_binary = uvs.tobytes()
    indices_binary = indices.tobytes()

    buffer_data = vertices_binary + normals_binary + uvs_binary + indices_binary
    buffer_length = len(buffer_data)

    # Create buffer views.
    vertices_buffer_view = BufferView(
        buffer=0,
        byteOffset=0,
        byteLength=len(vertices_binary),
        target=ARRAY_BUFFER,
    )

    normals_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertices_binary),
        byteLength=len(normals_binary),
        target=ARRAY_BUFFER,
    )

    uvs_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertices_binary) + len(normals_binary),
        byteLength=len(uvs_binary),
        target=ARRAY_BUFFER,
    )

    indices_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertices_binary) + len(normals_binary) + len(uvs_binary),
        byteLength=len(indices_binary),
        target=ELEMENT_ARRAY_BUFFER,
    )

    # Create accessors.
    vertices_accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=FLOAT,
        count=len(vertices),
        type="VEC3",
        min=vertices.min(axis=0).tolist(),
        max=vertices.max(axis=0).tolist(),
    )

    normals_accessor = Accessor(
        bufferView=1,
        byteOffset=0,
        componentType=FLOAT,
        count=len(normals),
        type="VEC3",
    )

    uvs_accessor = Accessor(
        bufferView=2,
        byteOffset=0,
        componentType=FLOAT,
        count=len(uvs),
        type="VEC2",
    )

    indices_accessor = Accessor(
        bufferView=3,
        byteOffset=0,
        componentType=UNSIGNED_INT,
        count=len(indices),
        type="SCALAR",
    )

    # Create PBR material.
    material = GltfMaterial(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=0),
            metallicRoughnessTexture=TextureInfo(index=2),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
        normalTexture=NormalMaterialTexture(index=1),
        doubleSided=False,
    )

    # Create textures and images.
    textures = [
        Texture(source=0),  # Color texture
        Texture(source=1),  # Normal texture
        Texture(source=2),  # Roughness texture
    ]

    images = [
        Image(uri=color_uri),
        Image(uri=normal_uri),
        Image(uri=roughness_uri),
    ]

    # Create mesh primitive.
    primitive = Primitive(
        attributes=Attributes(POSITION=0, NORMAL=1, TEXCOORD_0=2),
        indices=3,
        material=0,
    )

    mesh = Mesh(primitives=[primitive])

    # Create GLTF scene.
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[mesh],
        materials=[material],
        textures=textures,
        images=images,
        accessors=[
            vertices_accessor,
            normals_accessor,
            uvs_accessor,
            indices_accessor,
        ],
        bufferViews=[
            vertices_buffer_view,
            normals_buffer_view,
            uvs_buffer_view,
            indices_buffer_view,
        ],
        buffers=[Buffer(byteLength=buffer_length)],
    )

    # Set binary data.
    gltf.set_binary_blob(buffer_data)

    # Save GLTF file.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Suppress expected warning about binary blob conversion to .bin file.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*contains a binary blob.*", category=UserWarning
        )
        gltf.save(str(output_path))

    console_logger.info(
        f"Created 3D box GLTF: {output_path.name} "
        f"({width}m × {depth}m × {height}m at ({center_x}, {center_y}, {center_z}))"
    )


def create_wall_gltf(
    length: float,
    height: float,
    thickness: float,
    material: Material,
    output_path: Path,
    texture_scale: float = 0.5,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
    plane: Literal["XZ", "YZ"] = "XZ",
) -> None:
    """
    Create a 3D box wall mesh with PBR material.

    Args:
        length: Horizontal length dimension in meters.
        height: Vertical height dimension in meters.
        thickness: Wall thickness in meters.
        material: PBR material with textures.
        output_path: Where to save the GLTF file.
        texture_scale: Meters per texture tile (default: 0.5m).
        center_x: X position of wall center in Drake Z-up coordinates (default: 0.0).
        center_y: Y position of wall center in Drake Z-up coordinates (default: 0.0).
        center_z: Z position of wall center in Drake Z-up coordinates (default: 0.0).
        plane: "XZ" for front/back walls, "YZ" for left/right walls (default: "XZ").
    """
    if plane == "XZ":
        # Front/back walls: extend along X (length), thin in Y (thickness), tall in Z
        # (height).
        _create_box_mesh_gltf(
            width=length,
            depth=thickness,
            height=height,
            material=material,
            output_path=output_path,
            texture_scale=texture_scale,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
        )
    elif plane == "YZ":
        # Left/right walls: thin in X (thickness), extend along Y (length), tall in Z
        # (height).
        _create_box_mesh_gltf(
            width=thickness,
            depth=length,
            height=height,
            material=material,
            output_path=output_path,
            texture_scale=texture_scale,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
        )
    else:
        raise ValueError(f"Invalid plane: {plane}. Must be 'XZ' or 'YZ'.")


def create_floor_gltf(
    width: float,
    depth: float,
    thickness: float,
    material: Material,
    output_path: Path,
    texture_scale: float = 0.5,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> None:
    """
    Create a 3D box floor mesh with PBR material.

    Args:
        width: X dimension in meters.
        depth: Y dimension in meters.
        thickness: Floor thickness in meters (Z dimension).
        material: PBR material with textures.
        output_path: Where to save the GLTF file.
        texture_scale: Meters per texture tile (default: 0.5m).
        center_x: X position of floor center in Drake Z-up coordinates (default: 0.0).
        center_y: Y position of floor center in Drake Z-up coordinates (default: 0.0).
        center_z: Z position of floor center in Drake Z-up coordinates (default: 0.0).
    """
    _create_box_mesh_gltf(
        width=width,
        depth=depth,
        height=thickness,
        material=material,
        output_path=output_path,
        texture_scale=texture_scale,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
    )
