import pytest
from autoresearch.cosmos_window_timeline import publication_counts

def records(counts):
    cursor=0;out=[]
    for i,n in enumerate(counts):
        out.append({"index":i,"conditioning_frame":cursor,"executed_count":n});cursor+=n
    return out

@pytest.mark.parametrize("counts,expected",[([32,32],[32,33]),([32,14],[32,15]),([8,8,3],[8,8,4]),([1],[2])])
def test_timeline_keeps_actual_executed_prefix_and_terminal_frame(counts,expected):
    assert publication_counts(records(counts),sum(counts)+1)==expected

def test_timeline_rejects_mismatch_instead_of_splicing_unexecuted_future():
    with pytest.raises(ValueError):publication_counts(records([8,8]),65)
    r=records([8,8]);r[1]["conditioning_frame"]=32
    with pytest.raises(ValueError):publication_counts(r,17)
    for n in [0,True,33,8.]:
        with pytest.raises(ValueError):publication_counts([{"index":0,"conditioning_frame":0,"executed_count":n}],2)
