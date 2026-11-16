import io
import logging
import tempfile
import time

from pathlib import Path

import flask

from PIL import Image

from scenecode.agent_utils.blender.canonicalization import canonicalize_mesh_impl
from scenecode.agent_utils.blender.mesh_conversion import convert_glb_to_gltf_impl
from scenecode.agent_utils.blender.params import RenderParams
from scenecode.agent_utils.blender.render_dataclasses import LinkMeshInfo
from scenecode.agent_utils.blender.renderer import BlenderRenderer
from scenecode.agent_utils.house import ClearanceOpeningData

console_logger = logging.getLogger(__name__)


class BlenderRenderApp(flask.Flask):
    """The long-running Flask server application for Blender rendering."""

    def __init__(
        self,
        temp_dir: str,
        blend_file: Path | None = None,
        bpy_settings_file: Path | None = None,
    ) -> None:
        """Initialize the Flask app.

        Args:
            temp_dir: Temporary directory for storing intermediate files.
            blend_file: Optional path to a .blend file to use as base scene.
            bpy_settings_file: Optional path to a .py file with Blender settings.
        """
        super().__init__("scenecode_blender_render")

        self._temp_dir = temp_dir
        self._blender = BlenderRenderer(
            blend_file=blend_file,
            bpy_settings_file=bpy_settings_file,
        )

        # Storage for overlay rendering configuration.
        # NOTE: This is designed for single-threaded/sequential rendering use cases
        # where config is set once via /set_overlay_config before /render_overlay.
        # For concurrent rendering, consider thread-local storage or per-request config.
        self._overlay_config: dict[str, str | int] | None = None

        # Storage for blend export configuration.
        self._blend_config: dict[str, str] | None = None

        # Storage for floor plan rendering configuration.
        # NOTE: Same single-threaded assumption as overlay config.
        self._floor_plan_config: dict[str, str | int] | None = None

        self.add_url_rule("/", view_func=self._root_endpoint)

        # Configuration endpoint for overlay rendering.
        self.add_url_rule(
            rule="/set_overlay_config",
            endpoint="/set_overlay_config",
            methods=["POST"],
            view_func=self._set_overlay_config_endpoint,
        )

        # Standard rendering endpoint.
        endpoint = "/render"
        self.add_url_rule(
            rule=endpoint,
            endpoint=endpoint,
            methods=["POST"],
            view_func=self._render_endpoint,
        )

        # Overlay rendering endpoint.
        overlay_endpoint = "/render_overlay"
        self.add_url_rule(
            rule=overlay_endpoint,
            endpoint=overlay_endpoint,
            methods=["POST"],
            view_func=self._render_overlay_endpoint,
        )

        # Configuration endpoint for blend export.
        self.add_url_rule(
            rule="/set_blend_config",
            endpoint="/set_blend_config",
            methods=["POST"],
            view_func=self._set_blend_config_endpoint,
        )

        # Blend export endpoint.
        self.add_url_rule(
            rule="/save_blend",
            endpoint="/save_blend",
            methods=["POST"],
            view_func=self._save_blend_endpoint,
        )

        # Multiview rendering endpoint for VLM validation.
        self.add_url_rule(
            rule="/render_multiview",
            endpoint="/render_multiview",
            methods=["POST"],
            view_func=self._render_multiview_endpoint,
        )

        # Articulated multiview rendering endpoint for URDF/SDF conversion analysis.
        self.add_url_rule(
            rule="/render_multiview_articulated",
            endpoint="/render_multiview_articulated",
            methods=["POST"],
            view_func=self._render_multiview_articulated_endpoint,
        )

        # Configuration endpoint for floor plan rendering.
        self.add_url_rule(
            rule="/set_floor_plan_config",
            endpoint="/set_floor_plan_config",
            methods=["POST"],
            view_func=self._set_floor_plan_config_endpoint,
        )

        # Floor plan rendering endpoint (top-down view, no coordinate frame).
        self.add_url_rule(
            rule="/render_floor_plan",
            endpoint="/render_floor_plan",
            methods=["POST"],
            view_func=self._render_floor_plan_endpoint,
        )

        # Mesh canonicalization endpoint.
        self.add_url_rule(
            rule="/canonicalize",
            endpoint="/canonicalize",
            methods=["POST"],
            view_func=self._canonicalize_endpoint,
        )

        # GLB to GLTF conversion endpoint.
        self.add_url_rule(
            rule="/convert_glb_to_gltf",
            endpoint="/convert_glb_to_gltf",
            methods=["POST"],
            view_func=self._convert_glb_to_gltf_endpoint,
        )

    def _root_endpoint(self) -> str:
        """Display a banner page at the server root."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Scene Agent Blender Render Server</title>
        </head>
        <body>
            <h1>Scene Agent Blender Render Server</h1>
            <p>This server provides Blender-based rendering services for 
               scenecode.</p>
            <p>Send POST requests to /render to generate images.</p>
        </body>
        </html>
        """

    def _render_endpoint(self) -> flask.Response:
        """
        Accept a request to render and return the generated image (standard rendering).
        """
        try:
            # Parse the request parameters.
            params = self._parse_params(request=flask.request)

            # Render the scene using standard rendering.
            image_buffer = self._render(params)

            # Return the image as PNG response.
            return flask.send_file(
                image_buffer,
                mimetype="image/png",
                as_attachment=False,
                download_name="render.png",
            )

        except ValueError as e:
            flask.abort(400, description=str(e))
        except Exception as e:
            flask.abort(500, description=f"Rendering failed: {e}")

    def _parse_params(self, request: flask.Request) -> RenderParams:
        """Convert an HTTP request to a RenderParams object.

        Args:
            request: The Flask request object.

        Returns:
            The parsed render parameters.
        """
        # Save the uploaded scene file to temporary location.
        scene_file = request.files.get("scene")
        if not scene_file:
            raise ValueError("Missing scene file in request")

        temp_scene_path = Path(self._temp_dir) / "scene.gltf"
        scene_file.save(temp_scene_path)

        # Log GLB file size.
        file_size_mb = temp_scene_path.stat().st_size / (1024 * 1024)
        console_logger.info(f"GLB file received (size: {file_size_mb:.2f} MB)")

        # Extract form data.
        form = request.form

        # Parse required parameters.
        scene_sha256 = form.get("scene_sha256")
        if not scene_sha256:
            raise ValueError("Missing scene_sha256 parameter")

        image_type = form.get("image_type")
        if not image_type:
            raise ValueError("Missing image_type parameter")

        # Parse numeric parameters with validation.
        try:
            width = int(form.get("width", "640"))
            height = int(form.get("height", "480"))
            near = float(form.get("near", "0.1"))
            far = float(form.get("far", "100.0"))
            focal_x = float(form.get("focal_x", "500.0"))
            focal_y = float(form.get("focal_y", "500.0"))
            fov_x = float(form.get("fov_x", "1.047"))  # ~60 degrees.
            fov_y = float(form.get("fov_y", "0.785"))  # ~45 degrees.
            center_x = float(form.get("center_x", str(width / 2)))
            center_y = float(form.get("center_y", str(height / 2)))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric parameter: {e}")

        # Parse optional depth parameters.
        min_depth = None
        max_depth = None
        if image_type == "depth":
            min_depth_str = form.get("min_depth")
            max_depth_str = form.get("max_depth")
            if min_depth_str:
                min_depth = float(min_depth_str)
            if max_depth_str:
                max_depth = float(max_depth_str)

        return RenderParams(
            scene=temp_scene_path,
            scene_sha256=scene_sha256,
            image_type=image_type,
            width=width,
            height=height,
            near=near,
            far=far,
            focal_x=focal_x,
            focal_y=focal_y,
            fov_x=fov_x,
            fov_y=fov_y,
            center_x=center_x,
            center_y=center_y,
            min_depth=min_depth,
            max_depth=max_depth,
        )

    def _render(self, params: RenderParams) -> io.BytesIO:
        """Render the given scene, returning the PNG data buffer.

        Args:
            params: The rendering parameters.

        Returns:
            A BytesIO buffer containing the PNG image data.
        """
        # Create a temporary output file.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_output_path = Path(tmp_file.name)

        try:
            # Render the image using the Blender renderer.
            self._blender.render_image(params, temp_output_path)

            # Read the rendered image into a BytesIO buffer.
            image_data = temp_output_path.read_bytes()
            buffer = io.BytesIO(image_data)
            buffer.seek(0)

            return buffer
        finally:
            # Clean up the temporary file.
            if temp_output_path.exists():
                temp_output_path.unlink()

    def _set_overlay_config_endpoint(self) -> flask.Response:
        """Set configuration for overlay rendering.

        Expects JSON body with fields: output_dir, layout, top_view_width,
        top_view_height, side_view_count, side_view_width, side_view_height,
        scene_objects (optional), annotations (optional).

        Returns:
            JSON response with status.
        """
        try:
            config_data = flask.request.get_json()
            if not config_data:
                flask.abort(400, description="Missing JSON body")

            # Validate required fields.
            required_fields = [
                "output_dir",
                "layout",
                "top_view_width",
                "top_view_height",
                "side_view_count",
                "side_view_width",
                "side_view_height",
            ]
            for field in required_fields:
                if field not in config_data:
                    flask.abort(400, description=f"Missing required field: {field}")

            # Store the configuration (including optional scene_objects
            # and annotations).
            self._overlay_config = config_data
            scene_objects = config_data.get("scene_objects", [])

            # Configure TAA samples if provided (for ablation testing).
            taa_samples = config_data.get("taa_samples", 16)
            self._blender._taa_samples = taa_samples
            console_logger.debug(
                f"Overlay config set: layout={config_data['layout']}, "
                f"scene_objects={len(scene_objects)}"
            )
            for obj in scene_objects:
                has_bbox = obj.get("bounding_box") is not None
                console_logger.debug(f"  Object: {obj.get('name')}, bbox={has_bbox}")

            return flask.jsonify({"status": "success", "message": "Config stored"})

        except Exception as e:
            console_logger.error(f"Failed to set overlay config: {e}")
            flask.abort(500, description=f"Config setup failed: {e}")

    def _render_overlay_endpoint(self) -> flask.Response:
        """Render scene with overlays using stored configuration.

        Saves images to configured output_dir and returns a dummy PNG response
        (Drake requires PNG response, but actual images are in output_dir).

        Raises:
            RuntimeError: If overlay config has not been set.
        """
        request_start = time.time()
        console_logger.info("Overlay render request received")

        try:
            # Validate config is set.
            if self._overlay_config is None:
                flask.abort(
                    400,
                    description=(
                        "Overlay config not set. " "Call /set_overlay_config first"
                    ),
                )

            # Parse the request parameters.
            params = self._parse_params(request=flask.request)
            parse_time = time.time() - request_start
            console_logger.info(f"GLB file parsed in {parse_time:.2f} seconds")

            # Render the scene with overlays.
            render_start = time.time()
            console_logger.info("Starting overlay rendering")
            output_dir = Path(self._overlay_config["output_dir"])
            scene_objects = self._overlay_config["scene_objects"]
            annotations = self._overlay_config["annotations"]
            wall_normals = self._overlay_config["wall_normals"]
            # Support surfaces list for multi-surface rendering.
            support_surfaces = self._overlay_config.get("support_surfaces", None)
            show_support_surface = self._overlay_config.get(
                "show_support_surface", False
            )
            current_furniture_id = self._overlay_config.get(
                "current_furniture_id", None
            )
            # Context furniture IDs for manipuland mode (nearby furniture to keep visible).
            context_furniture_ids = self._overlay_config.get(
                "context_furniture_ids", None
            )
            # Single view mode for per-drawer rendering.
            render_single_view = self._overlay_config.get("render_single_view", None)
            # Wall surfaces for wall rendering modes.
            wall_surfaces = self._overlay_config.get("wall_surfaces", None)
            # Wall surfaces for top-down wall labels.
            wall_surfaces_for_labels = self._overlay_config.get(
                "wall_surfaces_for_labels", None
            )
            # Opening labels for door/window/opening visualization.
            # Convert dicts back to ClearanceOpeningData objects.
            openings_raw = self._overlay_config.get("openings", [])
            openings = [ClearanceOpeningData.from_dict(o) for o in openings_raw]
            # Ceiling perspective parameters.
            room_bounds = self._overlay_config.get("room_bounds", None)
            ceiling_height = self._overlay_config.get("ceiling_height", None)
            # Convert room_bounds list to tuple if provided.
            if room_bounds is not None:
                room_bounds = tuple(room_bounds)

            # Extract camera angle parameters for context image rendering.
            side_view_elevation_degrees = self._overlay_config.get(
                "side_view_elevation_degrees", None
            )
            side_view_start_azimuth_degrees = self._overlay_config.get(
                "side_view_start_azimuth_degrees", None
            )
            include_vertical_views = self._overlay_config.get(
                "include_vertical_views", True
            )

            image_paths = self._blender.render_agent_observation_views(
                params=params,
                output_dir=output_dir,
                layout=str(self._overlay_config["layout"]),
                top_view_width=int(self._overlay_config["top_view_width"]),
                top_view_height=int(self._overlay_config["top_view_height"]),
                side_view_count=int(self._overlay_config["side_view_count"]),
                side_view_width=int(self._overlay_config["side_view_width"]),
                side_view_height=int(self._overlay_config["side_view_height"]),
                scene_objects=scene_objects,
                annotations=annotations,
                wall_normals=wall_normals,
                support_surfaces=support_surfaces,
                show_support_surface=show_support_surface,
                current_furniture_id=current_furniture_id,
                context_furniture_ids=context_furniture_ids,
                render_single_view=render_single_view,
                openings=openings,
                wall_surfaces=wall_surfaces,
                wall_surfaces_for_labels=wall_surfaces_for_labels,
                room_bounds=room_bounds,
                ceiling_height=ceiling_height,
                side_view_elevation_degrees=side_view_elevation_degrees,
                side_view_start_azimuth_degrees=side_view_start_azimuth_degrees,
                include_vertical_views=include_vertical_views,
            )
            render_time = time.time() - render_start
            total_time = time.time() - request_start
            console_logger.info(
                f"Overlay rendering completed: {len(image_paths)} images "
                f"(render: {render_time:.2f}s, total: {total_time:.2f}s)"
            )

            # Return a dummy PNG matching Drake's CameraConfig dimensions.
            # Actual rendered images are already saved to output_dir.
            dummy_buffer = self._create_dummy_png(
                width=params.width, height=params.height
            )
            return flask.send_file(
                dummy_buffer,
                mimetype="image/png",
                as_attachment=False,
                download_name="overlay_render.png",
            )

        except ValueError as e:
            flask.abort(400, description=str(e))
        except Exception as e:
            flask.abort(500, description=f"Rendering failed: {e}")

    def _create_dummy_png(self, width: int, height: int) -> io.BytesIO:
        """Create a dummy RGBA PNG matching Drake's CameraConfig dimensions.

        Drake expects RGBA (4 channel) images matching the dimensions specified
        in CameraConfig. This creates a black RGBA PNG to satisfy that requirement.

        Args:
            width: Image width in pixels (from RenderParams/CameraConfig).
            height: Image height in pixels (from RenderParams/CameraConfig).

        Returns:
            BytesIO buffer containing a black RGBA PNG image.
        """
        # Create black RGBA image matching Drake's expected dimensions.
        img = Image.new("RGBA", (width, height), color=(0, 0, 0, 255))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    def _set_blend_config_endpoint(self) -> flask.Response:
        """Set configuration for .blend file export.

        Expects JSON body with field: output_path.

        Returns:
            JSON response with status.
        """
        try:
            config_data = flask.request.get_json()
            if not config_data:
                flask.abort(400, description="Missing JSON body")

            if "output_path" not in config_data:
                flask.abort(400, description="Missing required field: output_path")

            self._blend_config = config_data
            console_logger.debug(
                f"Blend config set: output_path={config_data['output_path']}"
            )

            return flask.jsonify(
                {"status": "success", "message": "Blend config stored"}
            )

        except Exception as e:
            console_logger.error(f"Failed to set blend config: {e}")
            flask.abort(500, description=f"Config setup failed: {e}")

    def _save_blend_endpoint(self) -> flask.Response:
        """Save scene as .blend file using stored configuration.

        Returns dummy PNG to satisfy Drake's HTTP contract.
        """
        request_start = time.time()
        console_logger.info("Save blend request received")

        try:
            if self._blend_config is None:
                flask.abort(
                    400,
                    description="Blend config not set. Call /set_blend_config first",
                )

            params = self._parse_params(request=flask.request)
            output_path = Path(self._blend_config["output_path"])

            self._blender.save_blend_file(params=params, output_path=output_path)

            total_time = time.time() - request_start
            console_logger.info(
                f"Blend file saved in {total_time:.2f}s to {output_path}"
            )

            # Return dummy PNG (Drake expects PNG response).
            dummy_buffer = self._create_dummy_png(
                width=params.width, height=params.height
            )
            return flask.send_file(
                dummy_buffer,
                mimetype="image/png",
                as_attachment=False,
                download_name="blend_export.png",
            )

        except ValueError as e:
            flask.abort(400, description=str(e))
        except Exception as e:
            console_logger.error(f"Failed to save blend file: {e}")
            flask.abort(500, description=f"Save blend failed: {e}")

    def _render_multiview_endpoint(self) -> flask.Response:
        """Render multiview images for VLM asset validation.

        This endpoint receives a mesh file and renders it from multiple views for
        validation. Unlike /render_overlay, this endpoint is self-contained and
        doesn't require pre-setting configuration - all parameters are passed
        in the request.

        Form data:
            mesh: The mesh file (GLB/GLTF) to render.
            output_dir: Directory where rendered images will be saved.
            num_side_views: Number of equidistant side views (default: 4).
            include_vertical_views: "true"/"false" for top/bottom views (default: true).
            width: Image width in pixels (default: 512).
            height: Image height in pixels (default: 512).
            light_energy: Light energy in watts (optional, uses default if not set).
            start_azimuth_degrees: Starting azimuth for side views (default: 0).
                Use 90 for wall-mounted objects where front is at +Y.
            show_coordinate_frame: "true"/"false" for RGB axes overlay (default: true).
                Set to "false" for cleaner validation renders.

        Returns:
            JSON response with status and list of rendered image paths.
        """
        request_start = time.time()
        console_logger.info("Multiview render request received")

        try:
            # Get mesh file from request.
            mesh_file = flask.request.files.get("mesh")
            if not mesh_file:
                flask.abort(400, description="Missing mesh file in request")

            # Save mesh to temp location.
            temp_mesh_path = Path(self._temp_dir) / "validation_mesh.glb"
            mesh_file.save(temp_mesh_path)

            file_size_kb = temp_mesh_path.stat().st_size / 1024
            console_logger.debug(f"Mesh file received ({file_size_kb:.1f} KB)")

            # Parse form parameters.
            form = flask.request.form

            output_dir_str = form.get("output_dir")
            if not output_dir_str:
                flask.abort(400, description="Missing output_dir parameter")
            output_dir = Path(output_dir_str)

            elevation_degrees_str = form.get("elevation_degrees")
            if not elevation_degrees_str:
                flask.abort(400, description="Missing elevation_degrees parameter")

            # Parse optional parameters with defaults.
            try:
                elevation_degrees = float(elevation_degrees_str)
                num_side_views = int(form.get("num_side_views", "4"))
                include_vertical = form.get("include_vertical_views", "true")
                include_vertical_views = include_vertical.lower() == "true"
                width = int(form.get("width", "512"))
                height = int(form.get("height", "512"))
                light_energy_str = form.get("light_energy")
                light_energy = float(light_energy_str) if light_energy_str else None
                start_azimuth_degrees = float(form.get("start_azimuth_degrees", "0.0"))
                show_coord_frame = form.get("show_coordinate_frame", "true")
                show_coordinate_frame = show_coord_frame.lower() == "true"
                taa_samples_str = form.get("taa_samples")
                taa_samples = int(taa_samples_str) if taa_samples_str else None
            except (ValueError, TypeError) as e:
                flask.abort(400, description=f"Invalid parameter: {e}")

            # Render using BlenderRenderer.
            render_start = time.time()
            # Build kwargs, only including taa_samples if provided.
            render_kwargs: dict = {
                "mesh_path": temp_mesh_path,
                "output_dir": output_dir,
                "elevation_degrees": elevation_degrees,
                "num_side_views": num_side_views,
                "include_vertical_views": include_vertical_views,
                "width": width,
                "height": height,
                "light_energy": light_energy,
                "start_azimuth_degrees": start_azimuth_degrees,
                "show_coordinate_frame": show_coordinate_frame,
            }
            if taa_samples is not None:
                render_kwargs["taa_samples"] = taa_samples
            image_paths = self._blender.render_multiview_for_analysis(**render_kwargs)
            render_time = time.time() - render_start
            total_time = time.time() - request_start

            console_logger.info(
                f"Multiview rendering completed: {len(image_paths)} images "
                f"(render: {render_time:.2f}s, total: {total_time:.2f}s)"
            )

            # Return paths as JSON.
            return flask.jsonify(
                {
                    "status": "success",
                    "image_paths": [str(p) for p in image_paths],
                }
            )

        except Exception as e:
            console_logger.error(f"Multiview rendering failed: {e}")
            flask.abort(500, description=f"Multiview render failed: {e}")

    def _render_multiview_articulated_endpoint(self) -> flask.Response:
        """Render multi-view images for articulated VLM physics analysis."""
        request_start = time.time()
        console_logger.info("Articulated multiview render request received")

        try:
            payload = flask.request.get_json()
            if not payload:
                flask.abort(400, description="Missing JSON body")

            link_mesh_payloads = payload.get("link_meshes")
            if not isinstance(link_mesh_payloads, list) or not link_mesh_payloads:
                flask.abort(400, description="Missing link_meshes")

            output_dir_str = payload.get("output_dir")
            if not output_dir_str:
                flask.abort(400, description="Missing output_dir")

            link_meshes = []
            try:
                for link_payload in link_mesh_payloads:
                    link_meshes.append(
                        LinkMeshInfo(
                            link_name=str(link_payload["link_name"]),
                            mesh_paths=[
                                Path(path) for path in link_payload.get("mesh_paths", [])
                            ],
                            origins=[
                                tuple(origin)
                                for origin in link_payload.get("origins", [])
                            ],
                            world_position=tuple(
                                link_payload.get("world_position", (0.0, 0.0, 0.0))
                            ),
                            world_rotation=(
                                tuple(tuple(row) for row in link_payload["world_rotation"])
                                if link_payload.get("world_rotation") is not None
                                else None
                            ),
                        )
                    )

                output_dir = Path(output_dir_str)
                num_combined_side_views = int(
                    payload.get("num_combined_side_views", 4)
                )
                num_link_side_views = int(payload.get("num_link_side_views", 4))
                width = int(payload.get("width", 512))
                height = int(payload.get("height", 512))
                light_energy = payload.get("light_energy")
                if light_energy is not None:
                    light_energy = float(light_energy)
                taa_samples = payload.get("taa_samples")
                if taa_samples is not None:
                    taa_samples = int(taa_samples)
            except (KeyError, TypeError, ValueError) as e:
                flask.abort(400, description=f"Invalid articulated render payload: {e}")

            render_start = time.time()
            render_result = self._blender.render_multiview_articulated(
                link_meshes=link_meshes,
                output_dir=output_dir,
                num_combined_side_views=num_combined_side_views,
                num_link_side_views=num_link_side_views,
                width=width,
                height=height,
                light_energy=light_energy,
                taa_samples=taa_samples,
            )
            render_time = time.time() - render_start
            total_time = time.time() - request_start
            total_images = len(render_result.combined_image_paths) + sum(
                len(paths) for paths in render_result.link_image_paths.values()
            )
            console_logger.info(
                f"Articulated multiview rendering completed: {total_images} images "
                f"(render: {render_time:.2f}s, total: {total_time:.2f}s)"
            )

            return flask.jsonify(
                {
                    "status": "success",
                    "combined_image_paths": [
                        str(path) for path in render_result.combined_image_paths
                    ],
                    "link_image_paths": {
                        link_name: [str(path) for path in paths]
                        for link_name, paths in render_result.link_image_paths.items()
                    },
                    "link_dimensions": {
                        link_name: list(dimensions)
                        for link_name, dimensions in render_result.link_dimensions.items()
                    },
                    "combined_dimensions": list(render_result.combined_dimensions),
                }
            )

        except Exception as e:
            console_logger.error(f"Articulated multiview rendering failed: {e}")
            flask.abort(500, description=f"Articulated multiview render failed: {e}")

    def _set_floor_plan_config_endpoint(self) -> flask.Response:
        """Set configuration for floor plan rendering.

        Expects JSON body with fields: output_path, width, height.
        Optional field: light_energy.

        Returns:
            JSON response with status.
        """
        try:
            config_data = flask.request.get_json()
            if not config_data:
                flask.abort(400, description="Missing JSON body")

            # Validate required fields.
            required_fields = ["output_path", "width", "height"]
            for field in required_fields:
                if field not in config_data:
                    flask.abort(400, description=f"Missing required field: {field}")

            # Store the configuration.
            self._floor_plan_config = config_data
            console_logger.debug(
                f"Floor plan config set: output_path={config_data['output_path']}, "
                f"width={config_data['width']}, height={config_data['height']}"
            )

            return flask.jsonify({"status": "success", "message": "Config stored"})

        except Exception as e:
            console_logger.error(f"Failed to set floor plan config: {e}")
            flask.abort(500, description=f"Config setup failed: {e}")

    def _render_floor_plan_endpoint(self) -> flask.Response:
        """Render a clean top-down floor plan view.

        Requires config to be set via /set_floor_plan_config before calling.
        Drake sends GLTF via RenderEngineGltfClient, and this endpoint renders
        to the configured output path, returning a dummy PNG to satisfy Drake.

        Returns:
            Dummy PNG (Drake requirement). Actual image saved to config path.
        """
        request_start = time.time()
        console_logger.info("Floor plan render request received")

        try:
            if self._floor_plan_config is None:
                flask.abort(
                    400,
                    description="Floor plan config not set. "
                    "Call /set_floor_plan_config first.",
                )

            # Parse GLTF from Drake request.
            params = self._parse_params(request=flask.request)
            parse_time = time.time() - request_start
            console_logger.info(f"GLTF file parsed in {parse_time:.2f} seconds")

            # Read config.
            output_path = Path(self._floor_plan_config["output_path"])
            width = int(self._floor_plan_config["width"])
            height = int(self._floor_plan_config["height"])
            light_energy = self._floor_plan_config.get("light_energy")

            # Render using BlenderRenderer.
            render_start = time.time()
            self._blender.render_floor_plan(
                mesh_path=params.scene,
                output_path=output_path,
                width=width,
                height=height,
                light_energy=light_energy,
            )
            render_time = time.time() - render_start
            total_time = time.time() - request_start

            console_logger.info(
                f"Floor plan rendering completed "
                f"(render: {render_time:.2f}s, total: {total_time:.2f}s)"
            )

            # Clear config after use.
            self._floor_plan_config = None

            # Return a dummy PNG matching Drake's CameraConfig dimensions.
            # Actual rendered image is already saved to output_path.
            dummy_buffer = self._create_dummy_png(
                width=params.width, height=params.height
            )
            return flask.send_file(
                dummy_buffer,
                mimetype="image/png",
                as_attachment=False,
                download_name="floor_plan_render.png",
            )

        except Exception as e:
            console_logger.error(f"Floor plan rendering failed: {e}")
            flask.abort(500, description=f"Floor plan render failed: {e}")

    def _canonicalize_endpoint(self) -> flask.Response:
        """Canonicalize mesh orientation and placement.

        This endpoint receives mesh parameters and canonicalizes the mesh to
        standard orientation using Blender. Unlike rendering endpoints, this
        is self-contained and doesn't require pre-set configuration.

        JSON body:
            input_path: Path to input GLTF file.
            output_path: Path where canonicalized GLTF will be saved.
            up_axis: Up axis in Blender coordinates (e.g., "+Z", "-Y").
            front_axis: Front axis in Blender coordinates (e.g., "+Y", "+X").
            object_type: Type of object (determines placement strategy).
                One of: "furniture", "manipuland", "wall_mounted", "ceiling_mounted".

        Returns:
            JSON response with status and output path.
        """
        request_start = time.time()
        console_logger.info("Canonicalize request received")

        try:
            # Parse JSON body.
            data = flask.request.get_json()
            if not data:
                flask.abort(400, description="Missing JSON body")

            # Validate required fields.
            required_fields = ["input_path", "output_path", "up_axis", "front_axis"]
            for field in required_fields:
                if field not in data:
                    flask.abort(400, description=f"Missing required field: {field}")

            input_path = Path(data["input_path"])
            output_path = Path(data["output_path"])
            up_axis = data["up_axis"]
            front_axis = data["front_axis"]
            object_type = data.get("object_type", "furniture")

            canonicalize_mesh_impl(
                input_path=input_path,
                output_path=output_path,
                up_axis=up_axis,
                front_axis=front_axis,
                object_type=object_type,
            )

            total_time = time.time() - request_start
            console_logger.info(
                f"Canonicalization completed in {total_time:.2f}s: {output_path}"
            )

            return flask.jsonify(
                {
                    "status": "success",
                    "output_path": str(output_path),
                }
            )

        except FileNotFoundError as e:
            console_logger.error(f"Canonicalization failed: {e}")
            flask.abort(404, description=str(e))
        except Exception as e:
            console_logger.error(f"Canonicalization failed: {e}")
            flask.abort(500, description=f"Canonicalization failed: {e}")

    def _convert_glb_to_gltf_endpoint(self) -> flask.Response:
        """Convert GLB file to GLTF with separate textures using Blender.

        This endpoint receives mesh parameters and converts the GLB to GLTF format
        with separate texture files. Running in the BlenderServer subprocess ensures
        bpy crashes don't kill the main scene worker process.

        JSON body:
            input_path: Path to input GLB or GLTF file.
            output_path: Path where converted GLTF will be saved.
            export_yup: If True, converts to Y-up GLTF standard. Default True.

        Returns:
            JSON response with status and output path.
        """
        request_start = time.time()
        console_logger.info("GLB to GLTF conversion request received")

        try:
            # Parse JSON body.
            data = flask.request.get_json()
            if not data:
                flask.abort(400, description="Missing JSON body")

            # Validate required fields.
            required_fields = ["input_path", "output_path"]
            for field in required_fields:
                if field not in data:
                    flask.abort(400, description=f"Missing required field: {field}")

            input_path = Path(data["input_path"])
            output_path = Path(data["output_path"])
            export_yup = data.get("export_yup", True)

            # Import and run bpy conversion (runs in this BlenderServer subprocess).
            convert_glb_to_gltf_impl(
                input_path=input_path,
                output_path=output_path,
                export_yup=export_yup,
            )

            total_time = time.time() - request_start
            console_logger.info(
                f"GLB to GLTF conversion completed in {total_time:.2f}s: {output_path}"
            )

            return flask.jsonify(
                {
                    "status": "success",
                    "output_path": str(output_path),
                }
            )

        except FileNotFoundError as e:
            console_logger.error(f"GLB to GLTF conversion failed: {e}")
            flask.abort(404, description=str(e))
        except Exception as e:
            console_logger.error(f"GLB to GLTF conversion failed: {e}")
            flask.abort(500, description=f"GLB to GLTF conversion failed: {e}")
