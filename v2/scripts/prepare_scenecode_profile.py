#!/usr/bin/env python3
"""Create immutable run-scoped SceneCode profiles bound to the Agent model."""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, stat, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORLD=ROOT/"code/WorldOlympiad-main"
FILES=("config.yaml","config_WallArt.yaml","config_simplemanip.yaml","config_structmanip.yaml")
SAFE=re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MODEL=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")

def stop(message: str) -> None: raise SystemExit(f"SceneCode profile preparation blocked: {message}")
def sha(data: bytes) -> str:return hashlib.sha256(data).hexdigest()
def load_config() -> dict:
    try:return json.loads((ROOT/"config/two-interface.json").read_text(encoding="utf-8"))
    except Exception as exc:stop(f"two-interface config is unreadable ({type(exc).__name__})")
def model_lines(path: Path, model: str) -> tuple[bytes,int]:
    raw=path.read_bytes(); text=raw.decode("utf-8"); output=[]; count=0
    for line in text.splitlines(keepends=True):
        body=line.rstrip("\r\n"); ending=line[len(body):]
        if body.lstrip().startswith("model_name:"):
            key_end=body.index("model_name:")+len("model_name:")
            output.append(body[:key_end]+" "+json.dumps(model)+ending);count+=1
        else:output.append(line)
    if count==0:stop(f"no model_name field found in {path.name}")
    return "".join(output).encode("utf-8"),count

def expected_profile(profile: Path, source: Path, model: str) -> dict:
    manifest_path=profile/"profile_manifest.json"
    try:data=json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:stop(f"existing profile receipt is unreadable ({type(exc).__name__}); preserved")
    if data.get("schema_version")!=1 or data.get("agent_model")!=model or data.get("source_root")!=str(source):
        stop("existing run profile was made from different settings; preserved")
    for name,record in data.get("files",{}).items():
        p=profile/name
        if not p.is_file() or sha(p.read_bytes())!=record.get("profile_sha256"):
            stop(f"existing profile differs from its receipt: {name}; preserved")
    if set(data.get("files",{}))!=set(FILES):stop("existing profile has an unexpected YAML set; preserved")
    return data

def validate_scene_contract(run_root: Path, profile: Path, model: str) -> str:
    python=ROOT/"envs/autoresearch/bin/python"
    if not python.is_file():stop("Agent/worker runtime is not prepared")
    env=os.environ.copy();env.update({"WO_WORKSPACE":str(ROOT),"WORLDOLYMPIAD_WORKSPACE":str(ROOT),"PYTHONPATH":str(WORLD)+os.pathsep+env.get("PYTHONPATH","")})
    code=("import pathlib,sys; from wo.tasks.scene import source_config_env; "
          "v=source_config_env(pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2])); "
          "print(v.get('SCENECODE_OPENAI_MODEL',''))")
    task=run_root/"tasks"/".profile-contract.json"
    result=subprocess.run([str(python),"-c",code,str(task),str(profile)],env=env,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=False)
    actual_model=result.stdout.strip()
    if result.returncode or actual_model!=model:
        stop("WorldOlympiad SceneCode source-profile contract rejected the run profile")
    return actual_model

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True);parser.add_argument("--goal-template",type=Path,required=True)
    parser.add_argument("--goal-kind",choices=("reference","evaluation"),default="reference")
    a=parser.parse_args()
    if not SAFE.fullmatch(a.run):stop("run id is not safe")
    config=load_config();model=str(config.get("agent_api",{}).get("model","")).strip()
    if not MODEL.fullmatch(model):stop("Agent API model identifier is missing or malformed")
    runtime=Path(config.get("tested_model",{}).get("scenecode_runtime","runtime/scenecode")).expanduser()
    source=(runtime if runtime.is_absolute() else ROOT/runtime).resolve()
    source=source/"scenecode/code_object/agent_framework"
    if not source.is_dir():stop(f"pinned SceneCode runtime profiles are missing: {source}")
    run_root=ROOT/"outputs/wo_runs"/a.run
    profile=run_root/"scenecode_profile"
    run_root.mkdir(mode=0o700,parents=True,exist_ok=True);os.chmod(run_root,0o700)
    if profile.exists() or profile.is_symlink():
        if profile.is_symlink() or not profile.is_dir():stop("run profile path is occupied; preserved")
        manifest=expected_profile(profile,source,model)
        print(f"Reused verified SceneCode profile for Agent model {model}: {profile}")
    else:
        staging=run_root/f".scenecode_profile.partial.{time.time_ns()}"
        staging.mkdir(mode=0o700)
        records={}
        for name in FILES:
            src=source/name
            if not src.is_file():stop(f"required SceneCode source profile is missing: {name}")
            raw,count=model_lines(src,model);dest=staging/name
            dest.write_bytes(raw);os.chmod(dest,0o600)
            records[name]={"source_sha256":sha(src.read_bytes()),"profile_sha256":sha(raw),"model_name_fields":count}
        manifest={"schema_version":1,"run":a.run,"agent_model":model,"source_root":str(source),"files":records}
        rec=staging/"profile_manifest.json";rec.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8");os.chmod(rec,0o600)
        if profile.exists() or profile.is_symlink():stop(f"profile path appeared; staged profile preserved at {staging}")
        os.rename(staging,profile)
        print(f"Prepared SceneCode profile for Agent model {model}: {profile}")
    actual_model=validate_scene_contract(run_root,profile,model)
    print(f"PASS: SceneCode source-profile model={actual_model}; matches Agent API model")
    template=a.goal_template.resolve()
    if not template.is_file():stop(f"reference goal template is missing: {template}")
    goal_dir=ROOT/"outputs/evaluator/goals";goal_dir.mkdir(mode=0o700,parents=True,exist_ok=True);os.chmod(goal_dir,0o700)
    goal=goal_dir/f"{a.run}_{a.goal_kind}.md"
    content=template.read_text(encoding="utf-8").rstrip()+"\n\n## Run-scoped SceneCode API profile\n\n"+(
      "When submitting the `scene` stage, include `source_config_root` in `args_json` with this exact run-local path: `"+str(profile)+"`. "
      "Do not omit the field. The four profile files are already prepared and bind SceneCode to the configured Agent model. "
      "Do not edit or replace the profile.\n")
    if goal.exists():
        if goal.read_text(encoding="utf-8")!=content:stop("generated run goal already exists with different contents; preserved")
    else:
        fd=os.open(goal,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w",encoding="utf-8") as f:f.write(content)
    print(f"Verified SceneCode source-profile contract; generated run goal: {goal}")
    return 0
if __name__=="__main__":raise SystemExit(main())
