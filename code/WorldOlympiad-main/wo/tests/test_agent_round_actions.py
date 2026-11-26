import asyncio
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from wo.agent import run as agent


def call(name, cid, args):
    return SimpleNamespace(type="tool_call_item", raw_item=SimpleNamespace(
        name=name, call_id=cid, arguments=json.dumps(args)))


def output(cid, value):
    return SimpleNamespace(type="tool_call_output_item", raw_item={"call_id": cid, "output": value})


def test_actual_test_and_job_returns_are_bound_to_their_call_even_out_of_order():
    items = [output("tests", "rc=0\n11 passed"),
             call("run_tests", "tests", {"paths":"wo/tests/test_real_boundary.py"}),
             call("submit", "job", {"kind":"scene_check", "args_json":json.dumps({"run":"dev","case":"a","api_key":"must-not-store"}), "after":"17"}),
             output("job", "queued job #18 (scene_check)")]
    actions = agent.round_actions(items)
    assert actions[0]["output"] == "rc=0\n11 passed"
    assert actions[0]["arguments"]["paths"] == "wo/tests/test_real_boundary.py"
    assert actions[1]["arguments"]["job_args"] == {"run":"dev","case":"a"}
    assert actions[1]["output"] == "queued job #18 (scene_check)"
    assert "must-not-store" not in json.dumps(actions)


def test_image_payload_and_edit_or_task_bodies_are_not_duplicated_in_logs():
    items = [call("view_image", "image", {"path":"/actual/frame.png"}),
             output("image", [{"type":"input_image","image_url":"data:image/png;base64,SECRETPIXELS"}]),
             call("edit_file", "edit", {"path":"wo/fix.py","old":"old code","new":"new code"}),
             output("edit", "edited wo/fix.py"),
             call("author_task", "task", {"run":"dev","task_spec_json":"private task body"}),
             output("task", "saved")]
    actions = agent.round_actions(items)
    encoded = json.dumps(actions)
    for content in ["SECRETPIXELS","old code","new code","private task body"]:
        assert content not in encoded
    assert actions[0]["arguments"]["path"] == "/actual/frame.png"
    assert actions[0]["output"] == {"image_payload_omitted":True}
    assert actions[1]["arguments"]["new_sha256"] == hashlib.sha256(b"new code").hexdigest()


def test_failed_validation_is_retained_and_unfinished_calls_are_not_invented():
    items = [call("run_tests","bad",{"paths":"wo/tests/test_contract.py"}),
             output("bad","rc=1\n1 failed"),call("promote","pending",{"message":"candidate"})]
    actions = agent.round_actions(items)
    assert actions[0]["output"].startswith("rc=1")
    assert "output" not in actions[1]


def test_trace_output_is_bounded_and_redacts_inline_credentials():
    token = "sk-" + "a" * 30
    items = [call("promote","c",{"message":"candidate"}),
             output("c", "Bearer abc.def " + token + "\n" + "x" * 10000)]
    text = agent.round_actions(items)[0]["output"]
    assert len(text) < 4100
    assert token not in text and "abc.def" not in text
    assert "trace shortened" in text


def test_one_round_persists_run_and_actual_actions_without_changing_dispatch(tmp_path, monkeypatch):
    items=[call("run_tests","t",{"paths":"wo/tests/test_contract.py"}),output("t","rc=0\n1 passed")]
    sdk=ModuleType("agents")
    sdk.Agent=lambda **kwargs: SimpleNamespace(**kwargs)
    class Runner:
        @staticmethod
        async def run(model, prompt, max_turns, run_config):
            assert max_turns == 60 and prompt == "current goal"
            return SimpleNamespace(final_output="validation pending",new_items=items)
    sdk.Runner=Runner;sdk.function_tool=lambda fn:fn
    monkeypatch.setitem(sys.modules,"agents",sdk)
    for name in ["scenecode","scenecode.utils"]:
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    config=ModuleType("scenecode.utils.openai");config.create_run_config=lambda **kwargs: None
    monkeypatch.setitem(sys.modules,"scenecode.utils.openai",config)
    monkeypatch.setattr(agent,"LOG",tmp_path)
    monkeypatch.setattr(agent,"round_input",lambda *args:"current goal")
    monkeypatch.setattr(agent,"refresh_tools",lambda:SimpleNamespace(ALL=[]))
    monkeypatch.setattr(agent.sys,"path",list(agent.sys.path))
    assert asyncio.run(agent.one_round("scope",Path("/goal.json"),7))=="validation pending"
    record=json.loads((tmp_path/"0007.json").read_text())
    assert record["run"]=="scope" and record["goal_path"]=="/goal.json"
    assert record["calls"]==["run_tests"]
    assert record["actions"][0]["output"]=="rc=0\n1 passed"
