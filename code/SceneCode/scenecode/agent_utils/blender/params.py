import dataclasses as dc

from pathlib import Path


@dc.dataclass
class RenderParams:
    """A dataclass that encapsulates all the necessary parameters to render a
    color, depth, or label image.

    https://drake.mit.edu/doxygen_cxx/group__render__engine__gltf__client__
    server__api.html#render-endpoint-form-data
    """

    scene: Path
    """The glTF input file."""

    scene_sha256: str
    """The checksum of `scene`."""

    image_type: str
    """The type of image being rendered (color, depth, or label)."""

    width: int
    """Width of the desired rendered image in pixels."""

    height: int
    """Height of the desired rendered image in pixels."""

    near: float
    """The near clipping plane of the camera as specified by the
    RenderCameraCore's ClippingRange::near() value."""

    far: float
    """The far clipping plane of the camera as specified by the
    RenderCameraCore's ClippingRange::far() value."""

    focal_x: float
    """The focal length x, in pixels, as specified by the
    systems::sensors::CameraInfo::focal_x() value."""

    focal_y: float
    """The focal length y, in pixels, as specified by the
    systems::sensors::CameraInfo::focal_y() value."""

    fov_x: float
    """The field of view in the x-direction (in radians) as specified by the
    systems::sensors::CameraInfo::fov_x() value."""

    fov_y: float
    """The field of view in the y-direction (in radians) as specified by the
    systems::sensors::CameraInfo::fov_y() value."""

    center_x: float
    """The principal point's x coordinate in pixels as specified by the
    systems::sensors::CameraInfo::center_x() value."""

    center_y: float
    """The principal point's y coordinate in pixels as specified by the
    systems::sensors::CameraInfo::center_y() value."""

    min_depth: float | None = None
    """The minimum depth range as specified by a depth sensor's
    DepthRange::min_depth(). Only provided when image_type="depth"."""

    max_depth: float | None = None
    """The maximum depth range as specified by a depth sensor's
    DepthRange::max_depth(). Only provided when image_type="depth"."""
