import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from wo.models import lingbot_va as rt


def make_runtime(tmp_path, monkeypatch, profile=None):
    monkeypatch.setattr(rt,'W',tmp_path)
    base=tmp_path/'base_python';base.write_bytes(b'base-python')
    monkeypatch.setattr(rt,'ROBOTWIN_PY',base)
    run=tmp_path/'outputs/wo_runs/run';out=run/'case/fastwam_s0'
    (out/'logs').mkdir(parents=True)
    manifest={'native_renderer':profile} if profile is not None else {}
    (run/'manifest.json').write_text(json.dumps(manifest))
    prefix=tmp_path/'runtime/robotwin_oidn/2.3.3';prefix.mkdir(parents=True)
    launcher=prefix/'python';launcher.write_bytes(b'isolated-launcher')
    library=prefix/'libOpenImageDenoise.so.2.3.3';library.write_bytes(b'reviewed-library')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (prefix/'runtime_receipt.json').write_text(json.dumps({
        'archive_sha256':'3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f',
        'launcher_sha256':digest(launcher),'library_sha256':{str(library):digest(library)}}))
    return run,out,library


@pytest.mark.parametrize('profile,expected',[ (None,'none'),('legacy_none','none'),('oidn233','oidn')])
def test_registered_profile_pins_renderer_and_preserves_scene(tmp_path,monkeypatch,profile,expected):
    run,out,library=make_runtime(tmp_path,monkeypatch,profile)
    monkeypatch.setenv('ROBOTWIN_RT_DENOISER','optix')
    scene=out/'scene_plan.json';scene.write_text(json.dumps({'position':[.1,.2,.3],'mass':.14,'friction':1.,'camera':{'fov':42}}))
    original=scene.read_bytes();calls=[]
    def execute(command,**kw):
        calls.append((command,kw['env']))
        native=out/'native_run';native.mkdir()
        settings={'camera_shader_dir':'/unchanged/rt','ray_tracing_samples_per_pixel':32,
                  'ray_tracing_path_depth':8,'ray_tracing_denoiser':kw['env']['ROBOTWIN_RT_DENOISER'],
                  'loaded_oidn_libraries':[str(library)] if profile=='oidn233' else []}
        (native/'renderer_config.json').write_text(json.dumps(settings))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(rt.subprocess,'run',execute)
    assert rt.run_client(out,['--plan',str(scene)],gpu=0)==0
    assert calls[0][1]['ROBOTWIN_RT_DENOISER']==expected
    assert scene.read_bytes()==original
    receipt=json.loads((out/'native_runtime.json').read_text())
    assert receipt['status']=='verified' and receipt['denoiser']==expected
    assert receipt['manifest_sha256']==hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest()
    assert receipt['launcher_sha256']
    if profile=='oidn233':
        assert receipt['dependency_receipt_sha256'] and receipt['actual_library_sha256']
        assert '/runtime/robotwin_oidn/2.3.3/python' in calls[0][0][0]
    else:
        assert calls[0][0][0]==str(rt.ROBOTWIN_PY)


def test_unknown_profile_is_rejected_before_native_launch(tmp_path,monkeypatch):
    _,out,_=make_runtime(tmp_path,monkeypatch,'unknown')
    monkeypatch.setattr(rt.subprocess,'run',lambda *a,**k:pytest.fail('invalid profile launched'))
    with pytest.raises(ValueError,match='unknown native_renderer'):
        rt.run_client(out,[],gpu=0)


def test_changed_launcher_or_dependency_is_rejected(tmp_path,monkeypatch):
    _,out,library=make_runtime(tmp_path,monkeypatch,'oidn233')
    library.write_bytes(b'changed')
    with pytest.raises(ValueError,match='native runtime file changed'):
        rt.native_runtime(out)


@pytest.mark.parametrize('field,value',[('ray_tracing_samples_per_pixel',16),('ray_tracing_path_depth',4),
                                       ('ray_tracing_denoiser','none'),('camera_shader_dir','/raster/default')])
def test_actual_settings_must_match_profile(tmp_path,monkeypatch,field,value):
    _,out,library=make_runtime(tmp_path,monkeypatch,'oidn233')
    def execute(command,**kw):
        native=out/'native_run';native.mkdir()
        settings={'camera_shader_dir':'/unchanged/rt','ray_tracing_samples_per_pixel':32,
                  'ray_tracing_path_depth':8,'ray_tracing_denoiser':'oidn','loaded_oidn_libraries':[str(library)]}
        settings[field]=value;(native/'renderer_config.json').write_text(json.dumps(settings))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(rt.subprocess,'run',execute)
    with pytest.raises(ValueError,match='actual native renderer differs'):
        rt.run_client(out,[],gpu=0)
    assert json.loads((out/'native_runtime.json').read_text())['status']=='failed'
