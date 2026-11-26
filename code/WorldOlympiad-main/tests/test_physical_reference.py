import json

import numpy as np
import pytest

from autoresearch.physical_reference import camera_at, summarize


def test_reference_only_summary_never_invents_model_score(tmp_path):
    root=tmp_path/'tasks/free_fall';root.mkdir(parents=True)
    (root/'simulation.json').write_text(json.dumps({'stage':'reference_simulated'}))
    summarize(tmp_path)
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['counts']=={'simulation':1,'render':0,'cae_frontend':0}
    assert result['model_comparison_score'] is None
    assert all(c['cae_frontend'] is None for c in result['cases'])
