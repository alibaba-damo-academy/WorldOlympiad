from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("wov2_preflight",ROOT/"scripts/preflight.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def card(index:int,free:int=32109,pids:list[int]|None=None,util:int=0,process_query_ok:bool=True)->dict:
    pids = list(pids or [])
    return {"index":index,"name":"NVIDIA A100","uuid":f"GPU-{index}",
            "total_mib":32607,"free_mib":free,"utilization":util,
            "compute_app_pids":pids,"compute_process_count":len(pids),
            "process_query_ok":process_query_ok}

def profile()->dict:
    return {"tested_model":{"adapter":"lingbot_va","gpu":"auto","policy_gpus":"auto","simulator_gpu":"auto"}}

def test_reference_preparation_accepts_one_idle_allowlisted_gpu():
    result=module.assess_resources({"cards":[card(1)]},"preparation",profile())
    assert result["ready"] is True
    assert result["selected_preparation_gpu_index"]==1
    assert result["required_preparation_gpu_count"]==1

def test_lingbot_evaluation_blocks_two_cards_without_three_card_topology():
    result=module.assess_resources({"cards":[card(1),card(2)]},"evaluation",profile())
    assert result["ready"] is False
    assert "two policy GPUs" in result["blocking_reason"]
    assert result["selected_policy_gpu_indices"]==[]
    assert result["selected_simulator_gpu_index"] is None

def test_lingbot_evaluation_selects_two_policy_and_distinct_simulator_cards():
    result=module.assess_resources({"cards":[card(0),card(1),card(2)]},"evaluation",profile())
    assert result["ready"] is True
    selected=set(result["selected_policy_gpu_indices"])
    assert len(selected)==2
    assert result["selected_simulator_gpu_index"] not in selected

def test_compute_process_pid_blocks_card_even_with_free_memory():
    result=module.assess_resources({"cards":[card(0,pids=[4321])]},"preparation",profile())
    assert result["ready"] is False
    assert result["eligible_preparation_gpu_indices"]==[]


def test_failed_compute_process_query_blocks_card():
    result=module.assess_resources({"cards":[card(0,process_query_ok=False)]},"preparation",profile())
    assert result["ready"] is False
    assert result["eligible_preparation_gpu_indices"]==[]
