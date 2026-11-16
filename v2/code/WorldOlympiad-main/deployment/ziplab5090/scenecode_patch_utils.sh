#!/usr/bin/env bash

# GNU patch may exit zero after applying only a subset of a multi-hunk patch.
# Call this on dry-run output before deciding that a compatibility patch fits.
scenecode_patch_output_has_rejects() {
  local output=${1:-}
  [[ $output == *FAILED* || $output == *ignored* || $output == *"malformed patch"* ]]
}
