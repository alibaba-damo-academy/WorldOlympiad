from copy import deepcopy
from types import SimpleNamespace
import pytest
from autoresearch.cosmos_evaluation_window import from_reference, validate_capture, digest


def test_success_does_not_shorten_a_shared_physical_time_window(tmp_path):
    path=tmp_path/'ref.json';path.write_text('{}')
    view=SimpleNamespace(frame_count=247,fps=15.,sha256='reference-video')
    ref=SimpleNamespace(view=lambda _name:view)
    window=from_reference(path,ref,'exterior_left')
    assert window['physical_duration_s']==16.4
    assert window['max_steps']==246
    for success in (True,False):
        metadata={'evaluation_window_sha256':digest(window),'termination_reason':'evaluation_window_complete',
                  'frame_count':247,'action_count':246,'task_success':success,'completion_frame':40 if success else None}
        validate_capture(metadata,window)
        short=deepcopy(metadata);short['frame_count']=41;short['action_count']=40
        with pytest.raises(ValueError,match='complete'):validate_capture(short,window)


def test_window_cannot_rescale_a_slow_prediction_to_hide_its_speed(tmp_path):
    path=tmp_path/'ref.json';path.write_text('{}')
    ref=SimpleNamespace(view=lambda _:SimpleNamespace(frame_count=31,fps=15.,sha256='video'))
    window=from_reference(path,ref,'exterior_left');window['model_time_rescaled']=True
    metadata={'evaluation_window_sha256':digest(window),'termination_reason':'evaluation_window_complete',
              'frame_count':31,'action_count':30,'task_success':False}
    with pytest.raises(ValueError,match='physical-time'):validate_capture(metadata,window)


def test_bound_window_cannot_be_dropped_or_injected(tmp_path):
    from autoresearch.cosmos_evaluation_window import validate_binding
    path=tmp_path/'ref.json';path.write_text('{}')
    window=from_reference(path,SimpleNamespace(view=lambda _:SimpleNamespace(frame_count=34,fps=15.,sha256='video')),'left')
    metadata={'evaluation_window':window,'evaluation_window_sha256':digest(window),
              'termination_reason':'evaluation_window_complete','frame_count':34,'action_count':33,
              'official_task_max_steps':33,'task_success':False,
              'windows':[{'planned_action_count':32,'executed_action_count':32},
                         {'planned_action_count':1,'executed_action_count':1}]}
    assert validate_binding(metadata,{'evaluation_window':window})==window
    with pytest.raises(ValueError,match='certified'):validate_binding(metadata,{})
    dropped=deepcopy(metadata);dropped.pop('evaluation_window')
    with pytest.raises(ValueError,match='certified'):validate_binding(dropped,{'evaluation_window':window})
    assert validate_binding({}, {}) is None
    partial=deepcopy(metadata);partial['windows'][0]['executed_action_count']=30
    with pytest.raises(ValueError,match='chunk early'):validate_binding(partial,{'evaluation_window':window})


@pytest.mark.parametrize('success',[True,False])
def test_output_manifest_does_not_confuse_window_completion_with_goal_success(success):
    from autoresearch.robotics_contracts import WAMOutputManifest
    payload=dict(schema_version='wam_output_bundle.v3',case_id='task',wam='cosmos3_nano_droid',
        source_model='nvidia/Cosmos3-Nano-Policy-DROID',checkpoint_revision='frozen',
        input_manifest='input_manifest.json',adapter_contract_sha256='a'*64,
        model_metadata='capture.json',raw_video='raw.mp4',selected_view='exterior_left',
        crop_xywh=(0,360,320,180),aligned_video='aligned.mp4',width=320,height=180,
        fps=15.,frame_count=247,duration_sec=247/15,termination_reason='evaluation_window_complete',
        task_success=success,conditioning_frame_mae_8bit=0.,quality_certificate='quality.json')
    assert WAMOutputManifest(**payload).task_success is success
    if success:
        payload['termination_reason']='native_task_timeout'
        with pytest.raises(ValueError,match='task success'):WAMOutputManifest(**payload)
