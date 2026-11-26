"""Score partially observed physics motion on common valid time intervals.

Complete pairs retain their independent native timelines and the original score.
For incomplete pairs, only intervals observed in both videos contribute. Native
samples are retained; interval boundaries are linearly evaluated inside observed
edges. Gaps contribute neither an invented displacement nor invented elapsed time.
"""
from pathlib import Path
import numpy as np
from .schema import ActionTrajectory
from .metrics import score_action_trajectories
from .current_metric import CURRENT_METRIC_VERSION
from .physics_object import object_trajectory, score_object_pair

PARTIAL_POLICY = "physics_common_observed_intervals.v1"

def visible_track(flow_path, fps, expected_frames):
    with np.load(flow_path, allow_pickle=False) as d:
        p = np.asarray(d["position"], dtype=float)
        mask = np.asarray(d["valid_mask"])
    if p.ndim != 3 or p.shape[-1] != 3 or len(p) != expected_frames or mask.shape != p.shape[:2]:
        raise ValueError("flow/video shape mismatch")
    if not np.isin(mask,[0,1]).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid validity/FPS")
    mask = mask.astype(bool) & np.isfinite(p).all(axis=2)
    base = mask[0]
    if base.sum() < 3:
        raise ValueError("initial object is not sufficiently observed")
    anchor = p[0,base].mean(0)
    positions = np.full((expected_frames,3),np.nan)
    observed = np.zeros(expected_frames,bool)
    for i in range(expected_frames):
        good = base & mask[i]
        if good.sum() < 3:
            continue
        x,y = p[0,good],p[i,good]
        if np.array_equal(x,y):
            positions[i] = anchor
            observed[i] = True
            continue
        xm,ym=x.mean(0),y.mean(0)
        if np.linalg.matrix_rank(x-xm)<2:
            continue
        u,_,vt=np.linalg.svd((x-xm).T@(y-ym))
        correction=np.eye(3);correction[2,2]=np.linalg.det(vt.T@u.T)
        rotation=vt.T@correction@u.T
        positions[i]=rotation@anchor+ym-rotation@xm
        observed[i]=True
    return positions,np.arange(expected_frames)/float(fps),observed

def support_intervals(times, observed):
    spans=[]
    for i in range(len(times)-1):
        if observed[i] and observed[i+1]:
            a,b=float(times[i]),float(times[i+1])
            if spans and abs(spans[-1][1]-a)<1e-10:
                spans[-1][1]=b
            else:
                spans.append([a,b])
    return spans

def intersect_support(left,right):
    i=j=0;out=[]
    while i<len(left) and j<len(right):
        a=max(left[i][0],right[j][0]);b=min(left[i][1],right[j][1])
        if b>a+1e-12:
            if out and abs(out[-1][1]-a)<1e-10:out[-1][1]=b
            else:out.append([a,b])
        if left[i][1]<=right[j][1]:i+=1
        else:j+=1
    return out

def observed_path(positions,times,observed,intervals,role):
    path=[];clock=[];end_position=np.zeros(3);end_time=0.
    def value(t):
        idx=int(np.searchsorted(times,t))
        if idx<len(times) and abs(times[idx]-t)<1e-10:
            if not observed[idx]:raise ValueError("unobserved boundary")
            return positions[idx]
        if idx==0 or idx==len(times) or not(observed[idx-1] and observed[idx]):
            raise ValueError("boundary would bridge a missing observation")
        weight=(t-times[idx-1])/(times[idx]-times[idx-1])
        return positions[idx-1]*(1-weight)+positions[idx]*weight
    for a,b in intervals:
        inside=times[(times>a+1e-10)&(times<b-1e-10)]
        ts=np.r_[a,inside,b]
        points=np.asarray([value(float(t)) for t in ts])
        relative=points-points[0]+end_position
        shifted=ts-a+end_time
        first=0 if not path else 1
        path.extend(relative[first:]);clock.extend(shifted[first:])
        end_position=relative[-1];end_time=float(shifted[-1])
    if len(path)<2:
        raise ValueError("no common observed motion interval")
    return ActionTrajectory(np.asarray(path),np.asarray(clock),
        coordinate_frame="registered_world_observed_displacements",
        metadata={"input_policy":PARTIAL_POLICY,"role":role,
                  "observed_frame_mask":[True]*len(path),
                  "source_observed_frame_mask":observed.tolist(),
                  "source_time_intervals_seconds":intervals,
                  "gap_displacement_inferred":False,"gap_duration_inferred":False})

def score_partial_pair(pair):
    roles=pair["roles"]
    complete={}
    try:
        for role,d in roles.items():
            complete[role]=object_trajectory(Path(d["dir"])/"object_flow_result.npz",
                fps=d["fps"],expected_frames=d["frames"],
                video_sha256=d["video_sha256"],role=role)
        result=score_object_pair(complete["reference"],complete["imagination"])
        result.update(partial_observation=False,reference_coverage=1.,prediction_coverage=1.,
                      coverage_policy=PARTIAL_POLICY,compared_intervals_seconds=None)
        return result,complete
    except ValueError:
        pass
    raw={role:visible_track(Path(d["dir"])/"object_flow_result.npz",d["fps"],d["frames"])
         for role,d in roles.items()}
    supports={role:support_intervals(t,m) for role,(_,t,m) in raw.items()}
    common=intersect_support(supports["reference"],supports["imagination"])
    if not common:
        raise ValueError("no jointly observed adjacent frames")
    trajectories={role:observed_path(p,t,m,common,role) for role,(p,t,m) in raw.items()}
    result=score_action_trajectories(trajectories["reference"],trajectories["imagination"]).to_dict()
    duration=sum(b-a for a,b in common)
    result.update(input_version=PARTIAL_POLICY,coverage_policy=PARTIAL_POLICY,
        partial_observation=True,observation_coverage="partial",
        measurement_scope="common observed displacement intervals only; not a complete robotics action episode",
        tracked_entity=pair.get("tracked_entity","primary task object"),
        compared_intervals_seconds=common,
        compared_observed_seconds=duration,
        reference_coverage=duration/float(raw["reference"][1][-1]),
        prediction_coverage=duration/float(raw["imagination"][1][-1]),
        reference_observation_coverage=sum(b-a for a,b in supports["reference"])/float(raw["reference"][1][-1]),
        prediction_observation_coverage=sum(b-a for a,b in supports["imagination"])/float(raw["imagination"][1][-1]))
    return result,trajectories
