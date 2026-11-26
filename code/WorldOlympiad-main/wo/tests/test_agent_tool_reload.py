import importlib
from pathlib import Path
import sys

from wo.agent.run import refresh_tools


def test_promoted_schema_and_tool_signature_replace_old_dispatcher(tmp_path, monkeypatch):
    pkg = tmp_path / "reload_probe"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    spec = pkg / "schema.py"
    tools = pkg / "tools.py"
    spec.write_text("class Task:\n version = 'old'\n")
    tools.write_text("from .schema import Task\ndef author_task():\n return Task.version\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    schema = importlib.import_module("reload_probe.schema")
    old_tools = importlib.import_module("reload_probe.tools")
    old_fn = old_tools.author_task
    assert old_fn() == "old"
    spec.write_text("class Task:\n version = 'promoted supports collection members'\n")
    tools.write_text("from .schema import Task\ndef author_task(members: int):\n return Task.version, members\n")
    refreshed = refresh_tools(schema, old_tools)
    assert refreshed.author_task is not old_fn
    assert refreshed.author_task(members=3) == ("promoted supports collection members", 3)
    for name in ("reload_probe.tools", "reload_probe.schema", "reload_probe"):
        sys.modules.pop(name, None)
