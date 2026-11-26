"""Versioned threshold profiles for benchmark certification."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import ThresholdProfile


PROFILE_DIRECTORY = Path(__file__).resolve().parent / "configs"
DEFAULT_PROFILE_ID = "scenecode_official_worldolympiad_bootstrap_v2"


def profile_path(profile_id: str = DEFAULT_PROFILE_ID) -> Path:
    if not profile_id or Path(profile_id).name != profile_id:
        raise ValueError("threshold profile id must be a plain filename stem")
    path = PROFILE_DIRECTORY / f"{profile_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"unknown threshold profile: {profile_id}")
    return path


def load_threshold_profile(profile_id: str = DEFAULT_PROFILE_ID) -> ThresholdProfile:
    path = profile_path(profile_id)
    profile = ThresholdProfile.model_validate_json(path.read_text(encoding="utf-8"))
    if profile.profile_id != profile_id:
        raise ValueError(
            f"threshold profile id mismatch: requested {profile_id}, file contains "
            f"{profile.profile_id}"
        )
    return profile


def threshold_profile_hash(profile: ThresholdProfile | str = DEFAULT_PROFILE_ID) -> str:
    value = load_threshold_profile(profile) if isinstance(profile, str) else profile
    return hashlib.sha256(value.canonical_json().encode("utf-8")).hexdigest()
