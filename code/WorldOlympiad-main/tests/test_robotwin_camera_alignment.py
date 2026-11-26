import numpy as np
import pytest
from autoresearch.robotwin_camera_alignment import upstream_resize_intrinsics


@pytest.mark.parametrize('target',[(320,256),(160,128)])
def test_projection_matches_upstream_resize_pixel_mapping(target):
    k=upstream_resize_intrinsics(320,240,*target,37)
    raw=upstream_resize_intrinsics(320,240,320,240,37)
    for x,y,z in [(0,0,1),(.1,.2,1.1),(-.3,-.1,2)]:
        u=raw['fx']*x/z+raw['cx'];v=raw['fy']*y/z+raw['cy']
        expected=[(u+.5)*target[0]/320-.5,(v+.5)*target[1]/240-.5]
        got=[k['fx']*x/z+k['cx'],k['fy']*y/z+k['cy']]
        assert got==pytest.approx(expected)
    assert k['fx']/k['fy']==pytest.approx(.9375)
