import ast,json
from pathlib import Path
from types import SimpleNamespace

def test_server_receipt_preserves_input_and_original_reset_result(tmp_path):
    source=Path(__file__).resolve().parents[1]/"scripts/launch_lingbot_va_server.py"
    tree=ast.parse(source.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="attested_reset")
    called=[]
    def reset(obj,prompt=None):called.append(prompt);return "upstream-result"
    episode=tmp_path/"episode";episode.mkdir()
    scope={"Path":Path,"json":json,"prompt_reset_base":reset,
        "wan_va_server":SimpleNamespace(prompt_clean=lambda text:"encoder-cleaned:"+text),
        "args":SimpleNamespace(config_name="robotwin",reset_seed=None,save_root=str(tmp_path))}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),scope)
    text="Open the left cabinet door using the left arm."
    result=scope["attested_reset"](SimpleNamespace(exp_save_root=str(episode)),prompt=text)
    assert result=="upstream-result" and called==[text]
    receipt=json.loads((episode/"prompt_receipt.json").read_text())
    assert receipt["raw_prompt"]==text
    assert receipt["encoder_prompt"]=="encoder-cleaned:"+text
    assert receipt["model_rng_policy"]=="upstream_unreset_random_stream"
    assert receipt["prompt_rewritten_by_adapter"] is False
    assert len((tmp_path/"received_prompts.jsonl").read_text().splitlines())==1
