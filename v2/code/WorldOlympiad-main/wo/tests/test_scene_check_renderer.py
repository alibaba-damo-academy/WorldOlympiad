import json
import sys
from types import SimpleNamespace

import pytest
from wo.models import scene_check
from wo.models.lingbot_va_client import renderer_settings


def test_renderer_settings_reads_native_getters(monkeypatch):
    expected={'camera_shader_dir':'/native/shaders/rt','ray_tracing_samples_per_pixel':32,
              'ray_tracing_path_depth':8,'ray_tracing_denoiser':'oidn'}
    render=SimpleNamespace(**{'get_'+k:lambda v=v:v for k,v in expected.items()})
    monkeypatch.setitem(sys.modules,'sapien',SimpleNamespace(render=render))
    actual=renderer_settings()
    assert {k:actual[k] for k in expected}==expected
    assert isinstance(actual['loaded_oidn_libraries'],list)


@pytest.mark.parametrize('log,config_present,accepted',[
    ('[svulkan2] [error] OIDN Error: unsupported device type: CUDA',True,False),
    ('[svulkan2] [error] OIDN Error: invalid handle',True,False),
    ('[svulkan2] [warning] renderer warning',True,True),
    ('',False,False)])
def test_scene_check_rejects_renderer_errors_even_with_success_exit_and_scene_audits(
        tmp_path,monkeypatch,log,config_present,accepted):
    out=tmp_path/'out'
    monkeypatch.setattr(scene_check.rt,'build_plan',lambda *a,**k:(
        {'required_visible':['object'],'support_representation':'native table'},None))
    def client(*args,**kwargs):
        native=out/'native_run';native.mkdir()
        (out/'logs/client.log').write_text(log)
        for name,content in [('dry_run.json',{}),('framing.json',{'outside':[]}),
                             ('input_audit.json',{'status':'ready'})]:
            (native/name).write_text(json.dumps(content))
        if config_present:
            (native/'renderer_config.json').write_text(json.dumps({'ray_tracing_denoiser':'oidn'}))
        return 0
    monkeypatch.setattr(scene_check.rt,'run_client',client)
    if accepted:
        result=scene_check.run(tmp_path/'ref',out)
        assert result['scene_ready'] and result['renderer']['status']=='ready'
    else:
        with pytest.raises(RuntimeError,match='native renderer check failed'):
            scene_check.run(tmp_path/'ref',out)
        assert not (out/'scene_ready.json').exists()
    audit=json.loads((out/'renderer_audit.json').read_text())
    assert audit['status']==('ready' if accepted else 'failed')
