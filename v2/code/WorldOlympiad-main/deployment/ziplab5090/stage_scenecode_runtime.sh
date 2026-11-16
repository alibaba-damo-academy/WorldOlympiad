#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
workspace=${WORLDOLYMPIAD_WORKSPACE:-$(cd -- "$script_dir/../../../.." && pwd -P)}
source_root=${SCENECODE_SOURCE_ROOT:-$workspace/code/SceneCode-main}
runtime_root=${SCENECODE_RUNTIME_ROOT:-$workspace/tmp/scenecode_runtime}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
marker=$runtime_root/.worldolympiad_scenecode_runtime_v1
source "$(dirname "${BASH_SOURCE[0]}")/scenecode_runtime_utils.sh"

if [[ ! -f "$source_root/main.py" ]]; then
  echo "SceneCode source is missing: $source_root" >&2
  exit 2
fi
if [[ "$source_root" == "$runtime_root" ]]; then
  echo "Runtime root must differ from the read-only SceneCode source root" >&2
  exit 2
fi
if [[ -e "$runtime_root" && ! -f "$marker" ]]; then
  if scenecode_runtime_is_interrupted_source_copy "$source_root" "$runtime_root"; then
    # A previous staging attempt may have stopped after copying the pinned
    # source and applying only some patches. Admit only an exact repository
    # identity match, then let the idempotent patch checks resume it.
    printf '%s\n' 'schema_version=scenecode_runtime.v1' \
      "source_root=$source_root" 'status=staging_recovered' > "$marker"
  else
    echo "Refusing to modify an unrecognized runtime directory: $runtime_root" >&2
    exit 2
  fi
fi

if [[ ! -e "$runtime_root" ]]; then
  mkdir -p "$runtime_root"
  cp -R "$source_root"/. "$runtime_root"/
  chmod -R u+w "$runtime_root"
  # Establish ownership before patching so an interrupted patch sequence is
  # recoverable on the next stage invocation.
  printf '%s\n' 'schema_version=scenecode_runtime.v1' \
    "source_root=$source_root" 'status=staging' > "$marker"
fi

# Blender runs with its own Python import path. Stage the dependency beside
# the server instead of assuming that the repository is importable there.
receipt_source="$world_repo/autoresearch/scenecode_mesh_export_receipt.py"
receipt_target="$runtime_root/scenecode/code_object/agent_framework/blender/_mesh_export_receipt.py"
if [[ ! -f "$receipt_source" ]]; then
  echo "Blender export receipt source is missing: $receipt_source" >&2
  exit 2
fi
if ! cmp -s "$receipt_source" "$receipt_target"; then
  cp "$receipt_source" "${receipt_target}.tmp.$$"
  mv "${receipt_target}.tmp.$$" "$receipt_target"
fi

patches=(
  "$world_repo/deployment/ziplab5090/patches/scenecode_restore_default_worker_context.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_inline_single_worker.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_floor_plan_observe_async.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_bwrap_writable_paths.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_critic_output_contract.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_gpt5_family_responses.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_code_object_gpt54.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_code_object_gpt55.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_code_object_gpt56_sol.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_text_asset_fallback.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_text_only_env.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_external_reference_image.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_bare_walls_env.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_bare_ceiling_env.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_optional_context_agents.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_conversion_retry.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_single_root_canonicalization.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_attached_parts.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_handle_attachment_coordinates.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_handle_partplan_center.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_preserve_required_manipulands.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_exact_required_counts.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_protect_pilot_manipulands.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_protect_cube_aliases.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_protect_required_postsettle.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_blender44_api_rules.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_color_validation_leniency.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_structural_support_validation_leniency.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_validation_render_framing.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_code_asset_validation_light.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_code_asset_validation_light_floor.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_solid_task_marker_fallback.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_solid_colored_mat_fallback.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_task_marker_extended_colors.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_thin_covering_validation_light.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_thin_covering_shape_words.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_validate_final_code_asset.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_atomic_action_log.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_async_openai_close.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_openai_transient_retry.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_code_object_llm_close.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_glb_export_artifact_check.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_skip_optional_invalid_support.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_lossless_manipuland_requirements.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_deterministic_structure_gate.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_deterministic_primitive_fallback.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_task_thin_covering_collision.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_task_planar_collision_origin.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_sdf_support_visual_meshes.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_codeobject_measured_bounds.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_bounded_robotics_loop.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_plan_issue_revision.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_object_prompt_semantics.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_support_anchor.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_code_object_sse.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_partial_asset_resume.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_part_bounds_contract.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_cached_part_recheck.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_tolerant_part_export.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_verified_mesh_frame.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_paired_mesh_export.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_virtual_massless_root.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_standalone_export_receipt.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_stage_failure_evidence.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_export_execution_evidence.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_cached_assembly_and_glb_fidelity.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_density_preserving_rescale.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_openai_image_responses_fallback.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_image_responses_retry.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_required_reference_images.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_image_asset_urdf_frames.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_node_link_lifetime.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_backend_startup_timeout_cleanup.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_curve_surface_export.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_object_requirements_bridge.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_exact_cavity_export.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_codegen_validation_feedback.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_bound_context_reference.patch"
  "$world_repo/deployment/ziplab5090/patches/scenecode_provided_object_reference.patch"
)
patch_present() {
  case $(basename "$1") in
    scenecode_restore_default_worker_context.patch)
      grep -Fq 'result_queue: multiprocessing.Queue = multiprocessing.Queue()' \
        "$runtime_root/scenecode/utils/parallel.py" \
        && ! grep -Fq 'multiprocessing.get_context(' \
          "$runtime_root/scenecode/utils/parallel.py" \
        && grep -Fq 'p = multiprocessing.Process(' \
          "$runtime_root/scenecode/utils/parallel.py" \
        && grep -Fq 'import bpy  # noqa: F401' "$runtime_root/main.py" \
        && ! grep -Fq '    import bpy  # noqa: F401' "$runtime_root/main.py"
      ;;
    scenecode_inline_single_worker.patch)
      grep -Fq 'SCENECODE_INLINE_SINGLE_WORKER' \
        "$runtime_root/scenecode/utils/parallel.py"
      ;;
    scenecode_floor_plan_observe_async.patch)
      grep -Fq 'async def observe_scene() -> list[ToolOutputImage | ToolOutputText]:' \
        "$runtime_root/scenecode/floor_plan_agents/tools/vision_tools.py"
      ;;
    scenecode_bwrap_writable_paths.patch)
      grep -Fq 'SCENECODE_BWRAP_WRITABLE_PATHS' \
        "$runtime_root/scenecode/agent_utils/blender/server_manager.py" \
        && grep -Fq 'bwrap_cmd.extend(["--bind", path_text, path_text])' \
          "$runtime_root/scenecode/agent_utils/blender/server_manager.py"
      ;;
    scenecode_provided_object_reference.patch)
      grep -Fq 'provided_reference = select_provided_object_reference(item)' \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py" \
        && grep -Fq 'record_provided_object_reference(output_dir, image_path, item, provided_reference)' \
          "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_bound_context_reference.patch)
      grep -Fq 'context_kind="furniture_context"' \
        "$runtime_root/scenecode/agent_utils/image_generation.py" \
        && grep -Fq 'context cache receipt does not bind this output path' \
          "$runtime_root/scenecode/agent_utils/image_generation.py" \
        && grep -Fq 'external context copy changed the expected output bytes' \
          "$runtime_root/scenecode/agent_utils/image_generation.py"
      ;;
    scenecode_critic_output_contract.patch)
      grep -Fq 'Final structured output contract:' \
        "$runtime_root/scenecode/agent_utils/base_stateful_agent.py" \
        && grep -Fq 'AgentOutputSchema(output_type).json_schema()' \
          "$runtime_root/scenecode/agent_utils/base_stateful_agent.py"
      ;;
    scenecode_gpt5_family_responses.patch)
      grep -Fq 'model.startswith("gpt-5.")' \
        "$runtime_root/scenecode/agent_utils/vlm_service.py"
      ;;
    scenecode_code_object_gpt54.patch)
      ! grep -REq '^[[:space:]]*model_name: "gemini-3.1-pro-preview"' \
        "$runtime_root/scenecode/code_object/agent_framework"/config*.yaml \
        && grep -Eq '^[[:space:]]*model_name: "gpt-5[.](4|5|6-sol)"' \
          "$runtime_root/scenecode/code_object/agent_framework/config.yaml"
      ;;
    scenecode_code_object_gpt55.patch)
      ! grep -REq '^[[:space:]]*model_name: "gpt-5[.]4"' \
        "$runtime_root/scenecode/code_object/agent_framework"/config*.yaml \
        && grep -Eq '^[[:space:]]*model_name: "gpt-5[.](5|6-sol)"' \
          "$runtime_root/scenecode/code_object/agent_framework/config.yaml"
      ;;
    scenecode_code_object_gpt56_sol.patch)
      ! grep -REq '^[[:space:]]*model_name: "gpt-5[.](4|5)"' \
        "$runtime_root/scenecode/code_object/agent_framework"/config*.yaml \
        && grep -Fq 'model_name: "gpt-5.6-sol"' \
          "$runtime_root/scenecode/code_object/agent_framework/config.yaml"
      ;;
    scenecode_text_asset_fallback.patch)
      grep -Fq 'def generate_from_text(' \
        "$runtime_root/scenecode/agent_utils/code_object_generation.py" \
        && grep -Fq 'using the vendored text-conditioned pipeline' \
          "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_text_only_env.patch)
      grep -Fq 'SCENECODE_CODE_OBJECT_TEXT_ONLY' \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_node_link_lifetime.patch)
      grep -Fq "Blender node-link lifetime and repeated execution" \
        "$runtime_root/scenecode/code_object/agent_framework/prompts/constructor.md" \
        && grep -Fq "Blender node-link lifetime and repeated execution" \
          "$runtime_root/scenecode/code_object/agent_framework/prompts/constructor_simplemanip.md" \
        && grep -Fq "Blender node-link lifetime and repeated execution" \
          "$runtime_root/scenecode/code_object/agent_framework/prompts/constructor_structmanip.md"
      ;;
    scenecode_backend_startup_timeout_cleanup.patch)
      grep -Fq 'start() still owns the shared Blender request lock here' \
        "$runtime_root/scenecode/code_object/agent_framework/blender/backend_manager.py" \
        && grep -Fq 'self._close_log_file()' \
          "$runtime_root/scenecode/code_object/agent_framework/blender/backend_manager.py" \
        && ! sed -n '/async def _wait_until_ready/,/async def stop(/p' \
          "$runtime_root/scenecode/code_object/agent_framework/blender/backend_manager.py" \
          | grep -Fq 'await self.stop()'
      ;;
    scenecode_required_reference_images.patch)
      grep -Fq "record_image_result(output_dir, result)" \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_image_asset_urdf_frames.patch)
      grep -Fq "result = prepare_articulated_image_result(output_dir, result)" \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_external_reference_image.patch)
      grep -Fq 'SCENECODE_CODE_OBJECT_REFERENCE_IMAGE' \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_bare_walls_env.patch)
      grep -Fq 'SCENECODE_BARE_WALLS=1: preserving the furnished scene' \
        "$runtime_root/scenecode/wall_agents/stateful_wall_agent.py"
      ;;
    scenecode_bare_ceiling_env.patch)
      grep -Fq 'SCENECODE_BARE_CEILING=1: preserving the scene' \
        "$runtime_root/scenecode/ceiling_agents/stateful_ceiling_agent.py"
      ;;
    scenecode_optional_context_agents.patch)
      grep -Fq 'Authored Optional-Context Policy' \
        "$runtime_root/scenecode/prompts/data/wall_agent/stateful_planner_agent.yaml" \
        && grep -Fq 'Authored Optional-Context Policy' \
          "$runtime_root/scenecode/prompts/data/ceiling_agent/stateful_planner_agent.yaml" \
        && grep -Fq 'Optional background assets have a fail-open omission policy' \
          "$runtime_root/scenecode/prompts/data/wall_agent/stateful_planner_runner_instruction.yaml" \
        && grep -Fq 'Optional background assets have a fail-open omission policy' \
          "$runtime_root/scenecode/prompts/data/ceiling_agent/stateful_planner_runner_instruction.yaml"
      ;;
    scenecode_conversion_retry.patch)
      grep -Fq 'Mesh conversion attempt %d/3 failed' \
        "$runtime_root/scenecode/agent_utils/asset_manager.py"
      ;;
    scenecode_single_root_canonicalization.patch)
      grep -Fq 'for obj in [root_obj, *root_obj.children_recursive]:' \
        "$runtime_root/scenecode/agent_utils/blender/canonicalization.py"
      ;;
    scenecode_attached_parts.patch)
      grep -Fq 'Attached parts must have positive geometric overlap' \
        "$runtime_root/scenecode/code_object/agent_framework/prompts/constructor_structmanip.md"
      ;;
    scenecode_handle_attachment_coordinates.patch)
      grep -Fq 'Cup-handle coordinates must be checked after applying the PartPlan position' \
        "$runtime_root/scenecode/code_object/agent_framework/prompts/constructor_structmanip.md"
      ;;
    scenecode_handle_partplan_center.patch)
      grep -Fq 'PartPlan handle positions are AABB centers, not host-wall coordinates' \
        "$runtime_root/scenecode/code_object/agent_framework/prompts/constructor_structmanip.md"
      ;;
    scenecode_preserve_required_manipulands.patch)
      grep -Fq 'Required-object preservation' \
        "$runtime_root/scenecode/prompts/data/manipuland_agent/designer_agent.yaml"
      ;;
    scenecode_exact_required_counts.patch)
      grep -Fq 'Preserve exact requested counts' \
        "$runtime_root/scenecode/prompts/data/manipuland_agent/designer_agent.yaml"
      ;;
    scenecode_protect_pilot_manipulands.patch)
      grep -Fq 'SCENECODE_PROTECT_PILOT_MANIPULANDS' \
        "$runtime_root/scenecode/manipuland_agents/tools/manipuland_tools.py"
      ;;
    scenecode_protect_cube_aliases.patch)
      grep -Fq '"toy_cube",' \
        "$runtime_root/scenecode/manipuland_agents/tools/manipuland_tools.py"
      ;;
    scenecode_protect_required_postsettle.patch)
      grep -Fq 'Preserving required pilot manipuland' \
        "$runtime_root/scenecode/agent_utils/physical_feasibility.py"
      ;;
    scenecode_blender44_api_rules.patch)
      grep -Fq 'Blender 4.4 API correctness (mandatory)' \
        "$runtime_root/scenecode/code_object/agent_framework/prompts/constructor_simplemanip.md" \
        && grep -Fq 'Blender 4.4 API correctness (mandatory)' \
          "$runtime_root/scenecode/code_object/agent_framework/prompts/constructor_structmanip.md"
      ;;
    scenecode_color_validation_leniency.patch)
      grep -Fq 'Validation studio lighting can wash saturated colors toward white' \
        "$runtime_root/scenecode/prompts/data/asset_router/asset_validation.yaml"
      ;;
    scenecode_structural_support_validation_leniency.patch)
      grep -Fq 'structural base, not an extra' \
        "$runtime_root/scenecode/prompts/data/asset_router/asset_validation.yaml"
      ;;
    scenecode_validation_render_framing.patch)
      grep -Fq 'The bounding sphere is conservative for every view' \
        "$runtime_root/scenecode/agent_utils/blender/renderer.py"
      ;;
    scenecode_code_asset_validation_light.patch)
      grep -Fq '25 W preserves hue while retaining enough edge contrast' \
          "$runtime_root/scenecode/agent_utils/asset_router/router.py" \
        || grep -Fq 'A measured 25 W floor made dark furniture nearly black' \
          "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_code_asset_validation_light_floor.patch)
      grep -Fq 'A measured 25 W floor made dark furniture nearly black' \
          "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_solid_task_marker_fallback.patch)
      grep -Fq 'Accepted deterministic solid-color task-marker fallback' \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_solid_colored_mat_fallback.patch)
      grep -Fq '"mat", "pad", "target"' \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_task_marker_extended_colors.patch)
      grep -Fq '"teal": (25, 155, 150)' \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_thin_covering_shape_words.patch)
      grep -Fq 'words & {"round", "circular"}' \
        "$runtime_root/scenecode/agent_utils/thin_covering_generator.py"
      ;;
    scenecode_thin_covering_validation_light.patch)
      grep -Fq 'exact marker hue remains observable' \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_validate_final_code_asset.patch)
      grep -Fq 'Final Code_Object candidates remain untrusted' \
          "$runtime_root/scenecode/agent_utils/asset_router/router.py" \
        && grep -Fq 'Rigid fallbacks are ordinary mesh candidates' \
          "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_atomic_action_log.patch)
      grep -Fq 'Refusing to discard corrupt action log' \
        "$runtime_root/scenecode/agent_utils/action_logger.py"
      ;;
    scenecode_async_openai_close.patch)
      grep -Fq 'await client.close()' \
        "$runtime_root/scenecode/agent_utils/turn_trimming_session.py" \
        && grep -Fq 'await client.close()' \
          "$runtime_root/scenecode/robot_eval/llm_utils.py" \
        && grep -Fq 'await client.close()' \
          "$runtime_root/scenecode/utils/openai.py"
      ;;
    scenecode_openai_transient_retry.patch)
      grep -Fq 'SCENECODE_OPENAI_MAX_RETRIES' \
        "$runtime_root/scenecode/utils/openai.py" \
        && grep -Fq 'max_retries=DEFAULT_OPENAI_MAX_RETRIES' \
          "$runtime_root/scenecode/utils/openai.py"
      ;;
    scenecode_code_object_llm_close.patch)
      grep -Fq 'async def _disconnect_llm_clients' \
        "$runtime_root/scenecode/code_object/agent_framework/pipeline.py" \
        && grep -Fq 'Close the lazily-created SDK client in its owning event loop.' \
          "$runtime_root/scenecode/code_object/agent_framework/utils/llm_client.py"
      ;;
    scenecode_glb_export_artifact_check.patch)
      grep -Fq 'accepted validated on-disk artifact' \
        "$runtime_root/scenecode/code_object/agent_framework/pipeline.py" \
        && grep -Fq 'int.from_bytes(header[8:12], "little") == file_size' \
          "$runtime_root/scenecode/code_object/agent_framework/pipeline.py"
      ;;
    scenecode_skip_optional_invalid_support.patch)
      grep -q 'Skipping optional furniture' \
        "$runtime_root/scenecode/manipuland_agents/stateful_manipuland_agent.py"
      ;;
    scenecode_lossless_manipuland_requirements.patch)
      grep -Fq 'Lossless requirement forwarding' \
        "$runtime_root/scenecode/prompts/data/manipuland_agent/analyze_furniture_for_placement.yaml"
      ;;
    scenecode_deterministic_structure_gate.patch)
      grep -Fq 'Deterministic WorldOlympiad structure gate' \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_deterministic_primitive_fallback.patch)
      grep -Fq 'Accepted deterministic solid %s fallback' \
        "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_codeobject_measured_bounds.patch)
      grep -Fq 'if not visual_dir.exists() or not any(visual_dir.glob("*.gltf")):' "$runtime_root/scenecode/agent_utils/urdf_to_sdf.py" \
        && grep -Fq 'Recomputed invalid CodeObject bounds' "$runtime_root/scenecode/agent_utils/room.py"
      ;;
    scenecode_sdf_support_visual_meshes.patch)
      grep -Fq '(sdf_dir / "blender_output" / "parts").is_dir()' "$runtime_root/scenecode/agent_utils/support_surface_extraction.py"
      ;;
    scenecode_task_planar_collision_origin.patch)
      grep -Fq 'is_wall=is_wall_covering' "$runtime_root/scenecode/agent_utils/asset_manager.py" \
        && grep -Fq 'y_center = depth / 2 if is_wall else 0.0' \
          "$runtime_root/scenecode/agent_utils/thin_covering_generator.py"
      ;;
    scenecode_task_thin_covering_collision.patch)
      grep -Fq 'is_task_planar_covering = request.object_type == ObjectType.MANIPULAND' \
        "$runtime_root/scenecode/agent_utils/asset_manager.py"
      ;;
    scenecode_plan_issue_revision.patch)
      grep -Fq 'if planner_agent and (result.corrections or actionable_issues):' "$runtime_root/scenecode/code_object/agent_framework/agents/planner_checker.py"
      ;;
    scenecode_bounded_robotics_loop.patch)
      grep -Fq "normalize_blender_code" "$runtime_root/scenecode/code_object/agent_framework/agents/part_constructor.py" \
        && grep -Fq "Never pay for a revision that cannot be checked" "$runtime_root/scenecode/code_object/agent_framework/agents/planner_checker.py" \
        && grep -Fq "refusing expensive construction/export" "$runtime_root/scenecode/code_object/agent_framework/pipeline.py"
      ;;
    scenecode_object_prompt_semantics.patch)
      grep -Fq "never force a non-furniture object" "$runtime_root/scenecode/code_object/agent_framework/agents/planner.py"
      ;;
    scenecode_support_anchor.patch)
      grep -Fq "asset_support_offset(original_asset)" "$runtime_root/scenecode/manipuland_agents/tools/manipuland_tools.py" \
        && grep -Fq "Keep surface-placed manipulands anchored" "$runtime_root/scenecode/agent_utils/rescale_helpers.py"
      ;;
    scenecode_code_object_sse.patch)
      grep -Fq "SCENECODE_CODE_OBJECT_STREAM" "$runtime_root/scenecode/code_object/agent_framework/utils/llm_client.py"
      ;;
    scenecode_partial_asset_resume.patch)
      grep -Fq "seed_plan(output_dir, text_input)" "$runtime_root/scenecode/agent_utils/code_object_generation.py" \
        && grep -Fq "seed_part(part, text_input, code_path)" "$runtime_root/scenecode/code_object/agent_framework/agents/part_constructor.py"
      ;;
    scenecode_curve_surface_export.patch)
      grep -Fq "snippets.append(curve_surface_export_code())" "$runtime_root/scenecode/code_object/agent_framework/pipeline.py" \
        && grep -Fq "snippets.append(curve_surface_export_code())" "$runtime_root/scenecode/code_object/agent_framework/pipeline_batch.py"
      ;;
    scenecode_part_bounds_contract.patch)
      grep -Fq "append_part_bounds_contract(normalize_blender_code(code), part.to_dict())" "$runtime_root/scenecode/code_object/agent_framework/agents/part_constructor.py"
      ;;
    scenecode_tolerant_part_export.patch)
      grep -Fq 'Construction incomplete; exporting the assembled parts' \
        "$runtime_root/scenecode/code_object/agent_framework/pipeline.py"
      ;;
    scenecode_cached_part_recheck.patch)
      grep -Fq 'initial_code=initial_code' "$runtime_root/scenecode/code_object/agent_framework/agents/part_constructor.py"
      ;;
    scenecode_verified_mesh_frame.patch)
      grep -Fq 'bind_visual_frame(urdf_visual, sdf_visual, urdf_dir)' "$runtime_root/scenecode/agent_utils/urdf_to_sdf.py"
      ;;
    scenecode_paired_mesh_export.patch)
      grep -Fq "record_blender_y_up_export(out_path)" "$runtime_root/scenecode/code_object/agent_framework/blender/headless_server.py"
      ;;
    scenecode_virtual_massless_root.patch)
      grep -Fq "normalize_virtual_roots(model)" "$runtime_root/scenecode/agent_utils/urdf_to_sdf.py"
      ;;
    scenecode_standalone_export_receipt.patch)
      grep -Fq 'record_blender_y_up_export = _load_export_receipt_writer()' "$runtime_root/scenecode/code_object/agent_framework/blender/headless_server.py"
      ;;
    scenecode_stage_failure_evidence.patch)
      grep -Fq '"stage_errors": self.stage_errors' "$runtime_root/scenecode/code_object/agent_framework/pipeline.py"
      ;;
    scenecode_export_execution_evidence.patch)
      grep -Fq '"error_type": type(exc).__name__' "$runtime_root/scenecode/code_object/agent_framework/blender/headless_server.py"
      ;;
    scenecode_cached_assembly_and_glb_fidelity.patch)
      grep -Fq 'Rechecking cached source through the measured per-part constructor' "$runtime_root/scenecode/code_object/agent_framework/pipeline.py"
      ;;
    scenecode_density_preserving_rescale.patch)
      grep -Fq 'A uniform geometric rescale preserves the material density' "$runtime_root/scenecode/agent_utils/sdf_generator.py"
      ;;
    scenecode_image_responses_retry.patch)
      grep -Fq "SCENECODE_IMAGE_RETRIES" "$runtime_root/scenecode/agent_utils/openai_images_fallback.py"
      ;;
    scenecode_openai_image_responses_fallback.patch)
      grep -Fq 'self.image_api = route_images_through_responses(self.client)' \
        "$runtime_root/scenecode/agent_utils/image_generation.py" \
        && grep -Fq 'def route_images_through_responses(' \
          "$runtime_root/scenecode/agent_utils/openai_images_fallback.py"
      ;;
    scenecode_object_requirements_bridge.patch)
      grep -Fq "asset_name=asset_name" "$runtime_root/scenecode/agent_utils/code_object_generation.py" \
        && grep -Fq "asset_name=item.short_name" "$runtime_root/scenecode/agent_utils/asset_router/router.py" \
        && grep -Fq 'tools=kwargs.get("tools")' "$runtime_root/scenecode/code_object/agent_framework/agents/base_agent.py" \
        && grep -Fq "prepare_request_context(output_dir, request_context)" "$runtime_root/scenecode/agent_utils/code_object_generation.py" \
        && grep -Fq "object_description=item.description" "$runtime_root/scenecode/agent_utils/asset_router/router.py" \
        && grep -Fq "contextual_messages(messages" "$runtime_root/scenecode/code_object/agent_framework/agents/base_agent.py"
      ;;
    scenecode_codegen_validation_feedback.patch)
      grep -Fq 'Previous generated candidate failed validation:' "$runtime_root/scenecode/agent_utils/asset_router/router.py" \
        && grep -Fq 'native_image_prompt += "\n\n" + repair_context' "$runtime_root/scenecode/agent_utils/asset_router/router.py" \
        && grep -Fq 'object_description=generation_description' "$runtime_root/scenecode/agent_utils/asset_router/router.py"
      ;;
    scenecode_exact_cavity_export.patch)
      grep -Fq "preserve_prismatic_cavities(sdf_path, sdf_path)" "$runtime_root/scenecode/agent_utils/asset_manager.py"
      ;;
    *) return 1 ;;
  esac
}
source "$world_repo/deployment/ziplab5090/scenecode_patch_utils.sh"
optional_patches=(
  scenecode_solid_task_marker_fallback.patch
  scenecode_solid_colored_mat_fallback.patch
  scenecode_task_marker_extended_colors.patch
)
is_optional_patch() {
  local candidate=${1##*/}
  local optional
  for optional in "${optional_patches[@]}"; do
    [[ $candidate == "$optional" ]] && return 0
  done
  return 1
}
for patch_file in "${patches[@]}"; do
  # Check the semantic marker first. GNU patch may accept a previously applied
  # prose-only hunk again with fuzz, duplicating prompt instructions while still
  # reporting a successful forward dry-run.
  if patch_present "$patch_file"; then
    continue
  fi
  dry_run_output=$(patch --dry-run --forward --batch -p1 -d "$runtime_root" < "$patch_file" 2>&1) || dry_run_output+=$'\npatch command returned non-zero'
  # GNU patch can return success when at least one hunk applied even though
  # other hunks were rejected. Never treat that partial application as valid.
  if ! scenecode_patch_output_has_rejects "$dry_run_output" && [[ $dry_run_output != *"patch command returned non-zero"* ]]; then
    apply_output=$(patch --forward --batch -p1 -d "$runtime_root" < "$patch_file" 2>&1) || {
      echo "$apply_output" >&2
      echo "SceneCode compatibility patch application failed: $patch_file" >&2
      exit 3
    }
    if scenecode_patch_output_has_rejects "$apply_output"; then
      echo "$apply_output" >&2
      echo "SceneCode compatibility patch was only partially applied: $patch_file" >&2
      exit 3
    fi
  elif is_optional_patch "$patch_file"; then
    echo "Skipping unsupported optional SceneCode patch (upstream source drift): ${patch_file##*/}" >&2
  else
    echo "$dry_run_output" >&2
    echo "SceneCode compatibility patch is neither applicable nor present: $patch_file" >&2
    exit 3
  fi
done

python_bin=${SCENECODE_PYTHON:-${SCENECODE_ENV:-$workspace/envs/scenecode}/bin/python}
if [[ -x "$python_bin" ]]; then
  "$python_bin" -m py_compile \
    "$runtime_root/scenecode/agent_utils/vlm_service.py" \
    "$runtime_root/scenecode/agent_utils/code_object_generation.py" \
    "$runtime_root/scenecode/agent_utils/asset_router/router.py" \
    "$runtime_root/scenecode/agent_utils/asset_manager.py" \
    "$runtime_root/scenecode/agent_utils/blender/canonicalization.py" \
    "$runtime_root/scenecode/agent_utils/blender/server_manager.py" \
    "$runtime_root/scenecode/utils/parallel.py" \
    "$runtime_root/main.py" \
    "$runtime_root/scenecode/agent_utils/blender/renderer.py" \
    "$runtime_root/scenecode/agent_utils/physical_feasibility.py" \
    "$runtime_root/scenecode/agent_utils/action_logger.py" \
    "$runtime_root/scenecode/agent_utils/turn_trimming_session.py" \
    "$runtime_root/scenecode/code_object/agent_framework/pipeline.py" \
    "$runtime_root/scenecode/code_object/agent_framework/utils/llm_client.py" \
    "$runtime_root/scenecode/manipuland_agents/tools/manipuland_tools.py" \
    "$runtime_root/scenecode/manipuland_agents/stateful_manipuland_agent.py" \
    "$runtime_root/scenecode/robot_eval/llm_utils.py" \
    "$runtime_root/scenecode/utils/openai.py" \
    "$runtime_root/scenecode/wall_agents/stateful_wall_agent.py" \
    "$runtime_root/scenecode/ceiling_agents/stateful_ceiling_agent.py" \
    "$runtime_root/scenecode/agent_utils/image_generation.py" \
    "$runtime_root/scenecode/agent_utils/openai_images_fallback.py"
fi

marker_tmp="${marker}.tmp.$$"
printf '%s\n' \
  'schema_version=scenecode_runtime.v1' \
  "source_root=$source_root" \
  'status=ready' \
  'patches=scenecode_provided_object_reference,scenecode_bound_context_reference,scenecode_restore_default_worker_context,scenecode_inline_single_worker,scenecode_floor_plan_observe_async,scenecode_bwrap_writable_paths,scenecode_critic_output_contract,scenecode_gpt5_family_responses,scenecode_code_object_gpt54,scenecode_code_object_gpt55,scenecode_code_object_gpt56_sol,scenecode_text_asset_fallback,scenecode_text_only_env,scenecode_bare_walls_env,scenecode_bare_ceiling_env,scenecode_optional_context_agents,scenecode_conversion_retry,scenecode_single_root_canonicalization,scenecode_attached_parts,scenecode_handle_attachment_coordinates,scenecode_handle_partplan_center,scenecode_preserve_required_manipulands,scenecode_exact_required_counts,scenecode_protect_pilot_manipulands,scenecode_protect_cube_aliases,scenecode_protect_required_postsettle,scenecode_blender44_api_rules,scenecode_color_validation_leniency,scenecode_validation_render_framing,scenecode_code_asset_validation_light,scenecode_solid_task_marker_fallback,scenecode_solid_colored_mat_fallback,scenecode_task_marker_extended_colors,scenecode_thin_covering_validation_light,scenecode_thin_covering_shape_words,scenecode_validate_final_code_asset,scenecode_atomic_action_log,scenecode_async_openai_close,scenecode_openai_transient_retry,scenecode_code_object_llm_close,scenecode_glb_export_artifact_check,scenecode_skip_optional_invalid_support,scenecode_lossless_manipuland_requirements,scenecode_deterministic_structure_gate,scenecode_deterministic_primitive_fallback,scenecode_task_thin_covering_collision,scenecode_task_planar_collision_origin,scenecode_sdf_support_visual_meshes,scenecode_codeobject_measured_bounds,scenecode_bounded_robotics_loop,scenecode_plan_issue_revision,scenecode_codegen_validation_feedback,scenecode_object_prompt_semantics,scenecode_support_anchor,scenecode_code_object_sse,scenecode_partial_asset_resume,scenecode_openai_image_responses_fallback,scenecode_image_responses_retry,scenecode_backend_startup_timeout_cleanup' \
  > "$marker_tmp"
mv "$marker_tmp" "$marker"
echo "$runtime_root"
