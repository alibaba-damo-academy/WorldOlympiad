#!/usr/bin/env bash

# Return success only when an unmarked runtime is demonstrably a copy of the
# configured SceneCode checkout. Patches may have modified tracked files, so
# compare repository identity (origin and commit), not a dirty-worktree check.
scenecode_runtime_is_interrupted_source_copy() {
  local source_root=$1 runtime_root=$2 source_head runtime_head source_origin runtime_origin
  [[ -d "$source_root/.git" && -d "$runtime_root/.git" ]] || return 1
  [[ -f "$source_root/main.py" && -f "$runtime_root/main.py" && \
     -d "$source_root/scenecode" && -d "$runtime_root/scenecode" ]] || return 1
  source_head=$(git -C "$source_root" rev-parse --verify HEAD 2>/dev/null) || return 1
  runtime_head=$(git -C "$runtime_root" rev-parse --verify HEAD 2>/dev/null) || return 1
  [[ -n "$source_head" && "$runtime_head" == "$source_head" ]] || return 1
  source_origin=$(git -C "$source_root" remote get-url origin 2>/dev/null) || return 1
  runtime_origin=$(git -C "$runtime_root" remote get-url origin 2>/dev/null) || return 1
  [[ -n "$source_origin" && "$runtime_origin" == "$source_origin" ]]
}
