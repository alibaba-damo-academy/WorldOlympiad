from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code/WorldOlympiad-main"))
os.environ["WO_WORKSPACE"]=str(ROOT)
from wo.agent.config import private_environment_path

def test_private_env_override_is_respected(tmp_path):
    env={"WO_AGENT_ENV_FILE":str(tmp_path/"outside"/"scenecode.env")}
    assert private_environment_path(env,tmp_path)==tmp_path/"outside"/"scenecode.env"

def test_private_env_defaults_to_workspace_secret_path(tmp_path):
    assert private_environment_path({},tmp_path)==tmp_path/".secrets"/"scenecode.env"
