from types import SimpleNamespace
import json
import pytest
from autoresearch.robotics_visual_context import visual_separation_evidence, reuse_identical_context_review
from autoresearch.io import sha256_file


def inputs():
    appearance={"passed":False,"checks":{"separated:panel":False,"separated:host":False},
                "pairs":[{"left":"host","right":"panel","left_pixels":120,"right_pixels":150,
                          "view_id":"exterior_left","separated":False}]}
    verdict={"objects_recognizable":{"host":True,"panel":True},
             "cross_object_separation":{"host|panel":"strong"},
             "separation_evidence":"The seam, handle and revealed cavity distinguish the moving panel."}
    components=[SimpleNamespace(component_id="panel",host_object_id="host")]
    return dict(appearance=appearance,verdict=verdict,vision_passed=True,
                components=components,host_visibility={"panel":{"passed":True}})


def test_same_colour_moving_part_needs_positive_structural_and_visibility_evidence():
    args=inputs();r=visual_separation_evidence(**args)
    assert r["passed"] and not r["raw_rgb_separation_passed"]
    assert len(r["structural_evidence"])==2
    assert args["appearance"]["passed"] is False


@pytest.mark.parametrize("failure",["no_review","weak","unrecognizable","hidden_host","no_pixels","no_explanation"])
def test_structural_claim_cannot_bypass_missing_evidence(failure):
    args=inputs()
    if failure=="no_review":args["vision_passed"]=False
    if failure=="weak":args["verdict"]["cross_object_separation"]["host|panel"]="weak"
    if failure=="unrecognizable":args["verdict"]["objects_recognizable"]["panel"]=False
    if failure=="hidden_host":args["host_visibility"]["panel"]["passed"]=False
    if failure=="no_pixels":args["appearance"]["pairs"][0]["left_pixels"]=0
    if failure=="no_explanation":args["verdict"]["separation_evidence"]=""
    assert not visual_separation_evidence(**args)["passed"]


def test_same_colour_independent_shapes_still_need_reviewed_distinction():
    args=inputs();args["components"]=[];args["host_visibility"]={}
    assert visual_separation_evidence(**args)["passed"]
    args["verdict"]["cross_object_separation"]["host|panel"]="none"
    assert not visual_separation_evidence(**args)["passed"]


def test_review_reuse_requires_same_pixels_prompt_and_model_and_keeps_negative_verdict(tmp_path):
    image=tmp_path/"montage.png";image.write_bytes(b"immutable rendered pixels")
    (tmp_path/"task_instruction.json").write_text(json.dumps({"instruction":"Open the door."}))
    source=tmp_path/"review.json"
    source.write_text(json.dumps({"vision_review":"required","vision_model":"review-model",
        "object_visual_descriptions":{"door":"gray door"},"object_roles":{"door":"manipuland"},
        "montage":{"sha256":sha256_file(image),"phases":["initial","final"],"views":["top"]},
        "vision_verdict":{"objects_recognizable":{"door":False}}}))
    args=dict(montage={"path":image,"phases":["initial","final"],"views":["top"]},
              instruction="Open the door.",descriptions={"door":"gray door"},
              roles={"door":"manipuland"},model="review-model")
    verdict,_=reuse_identical_context_review(source,**args)
    assert verdict["objects_recognizable"]["door"] is False
    with pytest.raises(ValueError,match="identical"):
        reuse_identical_context_review(source,**{**args,"instruction":"Close the door."})
    image.write_bytes(b"changed pixels")
    with pytest.raises(ValueError,match="identical"):
        reuse_identical_context_review(source,**args)
