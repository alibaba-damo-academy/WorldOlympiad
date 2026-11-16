"""Intra-turn image stripping filter for call_model_input_filter.

This module provides a filter that strips images from older observe_scene
outputs within a single turn, keeping only the last N observations with
images intact. This reduces token usage when agents call observe_scene
multiple times within a turn.

Context images in user messages are never touched - only function_call_output
items containing images (which must be from observe_scene, as no other tools
return images) are processed.
"""

import logging
import re

from dataclasses import dataclass
from typing import Any

from agents.items import TResponseInputItem
from agents.run import CallModelData, ModelInputData
from omegaconf import DictConfig

from scenecode.agent_utils.turn_trimming_session import _is_image_content

console_logger = logging.getLogger(__name__)


@dataclass
class IntraTurnImageFilter:
    """Callable filter for stripping images from older observations within a turn.

    This filter is designed to be used with RunConfig.call_model_input_filter.
    It processes the input items before each model call, stripping images from
    observe_scene outputs beyond the configured threshold while preserving
    context images in user messages.

    Detection Strategy:
    - observe_scene outputs: type="function_call_output" with output containing
      image content (type="input_image" or similar)
    - context images: role="user" messages with content containing input_image
      (NOT stripped - these are essential reference images)

    Optimization:
    - Caches stripped items and reuses them when no new observations are added.
    - Only runs stripping logic when the last item is a new observation.
    """

    cfg: DictConfig
    _cached_items: list[TResponseInputItem] | None = None
    _cached_input_len: int = 0

    def __call__(self, data: CallModelData[Any]) -> ModelInputData:
        """Filter model input to strip images from older observations.

        Args:
            data: CallModelData containing agent, context, and model input.

        Returns:
            ModelInputData with possibly modified input items.
        """
        intra_cfg = self.cfg.session_memory.intra_turn_observation_stripping
        if not intra_cfg.enabled:
            return data.model_data

        input_items = list(data.model_data.input)
        current_len = len(input_items)

        # Optimization: if we have cached items and no new observation was added,
        # reuse the cached result. This avoids redundant stripping on every call.
        if self._cached_items is not None and current_len > self._cached_input_len:
            # Check if new items (since cache) contain an observation.
            new_items = input_items[self._cached_input_len :]
            has_new_observation = any(
                self._is_observation_output(item) for item in new_items
            )
            if not has_new_observation:
                # No new observation, extend cache with new items and return.
                self._cached_items = self._cached_items + new_items
                self._cached_input_len = current_len
                return ModelInputData(
                    input=self._cached_items, instructions=data.model_data.instructions
                )

        # Need to run full stripping logic.
        keep_n = intra_cfg.keep_last_n_observations
        items = list(input_items)  # Copy to avoid mutation.

        # Find all observation output indices (function_call_output with images).
        obs_indices = [
            i for i, item in enumerate(items) if self._is_observation_output(item)
        ]

        # Keep last N, strip older ones.
        if len(obs_indices) > keep_n:
            indices_to_strip = set(obs_indices[:-keep_n])
            n_stripped = len(indices_to_strip)
            for i in indices_to_strip:
                items[i] = self._strip_images_from_output(items[i])
            console_logger.info(
                f"Intra-turn stripping: {len(obs_indices)} observations found, "
                f"keeping last {keep_n}, stripped images from {n_stripped}"
            )

        # Update cache.
        self._cached_items = items
        self._cached_input_len = current_len

        return ModelInputData(input=items, instructions=data.model_data.instructions)

    def _is_observation_output(self, item: TResponseInputItem) -> bool:
        """Check if item is a function_call_output containing images.

        Only observe_scene returns images via tool outputs. All other tools
        return text only. So any function_call_output with images must be
        from observe_scene.

        Args:
            item: Input item to check.

        Returns:
            True if this is a function_call_output containing images.
        """
        if not isinstance(item, dict):
            return False
        if item.get("type") != "function_call_output":
            return False

        output = item.get("output")
        if isinstance(output, list):
            # Check if any element is an image.
            return any(isinstance(p, dict) and _is_image_content(p) for p in output)
        elif isinstance(output, str):
            # Check for embedded base64 (rare case).
            return "data:image/" in output and "base64," in output

        return False

    def _strip_images_from_output(self, item: TResponseInputItem) -> TResponseInputItem:
        """Replace images in output with placeholder text.

        Args:
            item: function_call_output item containing images.

        Returns:
            Copy of item with images replaced by placeholders.
        """
        result = dict(item)
        output = item.get("output")

        if isinstance(output, list):
            result["output"] = [
                (
                    {"type": "input_text", "text": "[image removed]"}
                    if isinstance(p, dict) and _is_image_content(p)
                    else p
                )
                for p in output
            ]
        elif isinstance(output, str) and "data:image/" in output:
            result["output"] = re.sub(
                r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
                "[base64 image removed]",
                output,
            )

        return result
