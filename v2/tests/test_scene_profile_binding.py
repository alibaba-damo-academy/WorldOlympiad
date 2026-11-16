from __future__ import annotations
import os
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
WORLD=ROOT/"code/WorldOlympiad-main"
sys.path.insert(0,str(WORLD))
os.environ["WO_WORKSPACE"]=str(ROOT)
from wo.agent.stages import bind_scene_source_profile

def test_scene_profile_is_added_inside_the_run(tmp_path,monkeypatch):
    run="profile_contract_case"
    expected=ROOT/"outputs/wo_runs"/run/"scenecode_profile"
    monkeypatch.setenv("WO_SCENECODE_SOURCE_CONFIG_ROOT",str(expected))
    value=bind_scene_source_profile({"run":run,"case":"case"})
    assert value["source_config_root"]==str(expected)

def test_conflicting_scene_profile_is_rejected(monkeypatch,tmp_path):
    run="profile_conflict_case"
    expected=ROOT/"outputs/wo_runs"/run/"scenecode_profile"
    monkeypatch.setenv("WO_SCENECODE_SOURCE_CONFIG_ROOT",str(expected))
    with pytest.raises(ValueError,match="differs"):
        bind_scene_source_profile({"run":run,"source_config_root":str(tmp_path/"other")})

def test_scene_profile_outside_run_is_rejected(monkeypatch,tmp_path):
    monkeypatch.setenv("WO_SCENECODE_SOURCE_CONFIG_ROOT",str(tmp_path/"profile"))
    with pytest.raises(ValueError,match="inside this run"):
        bind_scene_source_profile({"run":"outside_profile_case"})
