"""CAE v2 for native WAM episodes from their simulator-executed trajectories.

    python -m wo.eval.cae_episodes EPISODE_DIR [EPISODE_DIR ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

from .cae_v2 import Track, score
from .reference_binding import verify_binding
from ..render_plan import _FK, _T, _quat_xyzw, _prepared_urdf
from ..scene import ROBOT_URDF

_FKC = None


_ART_FK: dict = {}


def articulation_urdf(ref: Path, host: str) -> Path:
    """The joint-preserving URDF the expert simulated for an articulated host."""
    scene_path = ref / "scene_inputs.json"
    if scene_path.is_file():
        scene = json.loads(scene_path.read_text())
        objects = {scene["support"]["object_id"]: scene["support"], **scene["objects"]}
        return _prepared_urdf(ref, host, objects[host])
    found = sorted((ref / "physics_sdf").glob(f"{host}_s*/*.urdf"))
    if not found:
        raise FileNotFoundError(f"{ref}: no articulated URDF for {host}")
    if len(found) != 1:
        raise ValueError(f"{ref}: ambiguous legacy articulated URDF for {host}")
    return found[0]


def moving_part_position(urdf: Path, host_state: dict, joint: str, child: str, q: float) -> np.ndarray:
    """World centre of mass of the moving part (child link) at joint value q.

    Both the reference and the model episode are measured through the same
    forward kinematics of the same asset, so only the joint value is compared.
    (The child link frame of a hinge sits on the axis and would not move.)
    """
    fk = _ART_FK.get(str(urdf)) or _ART_FK.setdefault(str(urdf), _FK(urdf, fixed=True))
    if joint not in fk.joint or child not in fk.link_index:
        raise ValueError(f"articulation binding {joint}/{child} is absent from {urdf}")
    base = _T(host_state["link_position"], _quat_xyzw(host_state["link_orientation_xyzw"]))
    fk.pose(base, {joint: q})
    state = fk.p.getLinkState(fk.body, fk.link_index[child], computeForwardKinematics=True,
                              physicsClientId=fk.cid)
    return np.asarray(state[0], dtype=float)


def entity_tracks(traj_path: Path, objects: list[str], articulations: list[dict] = (),
                  ref: Path | None = None) -> dict[str, Track]:
    """Fingertip TCP (forward kinematics) + named object link positions.

    Articulated components (moving parts) are tracked by the centre of mass of
    their child link, from the recorded joint value (articulation_positions).
    """
    global _FKC
    _FKC = _FKC or _FK(ROBOT_URDF, fixed=True)
    frames = json.loads(Path(traj_path).read_text())["frames"]
    t = np.array([f.get("time_s", i / 15.0) for i, f in enumerate(frames)], dtype=float)
    rb = frames[0]["robot"]
    base = _T(rb["base_position"], _quat_xyzw(rb["base_orientation_xyzw"]))
    # Franka (DROID) episodes: fingertip TCP by forward kinematics of the shared
    # URDF.  Other embodiments (e.g. LingBot-VA's aloha arms) record their TCP.
    recorded = "end_effector_position" in rb and not set(rb["joint_names"]) & set(_FKC.joint)
    tcp = []
    for f in frames:
        r = f["robot"]
        if recorded:
            tcp.append(np.asarray(r["end_effector_position"], dtype=float))
            continue
        M = _FKC.pose(base, dict(zip(r["joint_names"], r["joint_positions"])))["base_link"]
        tcp.append(M[:3, 3] + M[:3, :3] @ np.array([0, 0, 0.15]))
    out = {"end_effector": Track(np.array(tcp), t, "end_effector")}
    for oid in objects:
        if all(oid in f.get("objects", {}) for f in frames):
            out[oid] = Track(np.array([f["objects"][oid]["link_position"] for f in frames]), t)
    for a in articulations:
        comp = a["component"]
        if comp in objects and ref is not None and all(
                comp in (f.get("articulation_positions") or {}) for f in frames):
            urdf = articulation_urdf(ref, a["host"])
            out[comp] = Track(np.array([moving_part_position(
                urdf, f["objects"][a["host"]], a["joint_name"], a["child_link"],
                f["articulation_positions"][comp]) for f in frames]), t)
    return out


def episode_articulations(graph: dict, ref: Path) -> list[dict]:
    from ..scene import resolved_articulations
    cert = json.loads((ref / "certificate.json").read_text())
    return resolved_articulations(cert, graph.get("articulation_components") or [])


def require_entity_tracks(tracks: dict, entities: list[str], source: str) -> None:
    missing = [name for name in entities if name not in tracks or len(tracks[name].positions) < 2]
    if missing:
        raise ValueError(f"{source}: missing required entity tracks: {', '.join(missing)}")


def _capture_metadata(ep_dir: Path, episode: dict) -> dict:
    """Read native capture evidence from supported adapter output layouts.

    Most reference-bound adapters write capture metadata beside ``capture/``;
    stock-task adapters may put it below ``native_run/capture/``. The embedded
    receipt is a final provenance-preserving fallback for older episodes.
    """
    for relative in ("native_run/capture/capture_metadata.json",
                     "capture/capture_metadata.json"):
        path = ep_dir / relative
        if path.is_file():
            return json.loads(path.read_text())
    embedded = episode.get("native_result")
    if isinstance(embedded, dict):
        return embedded
    raise FileNotFoundError(
        f"episode has no native capture metadata in supported locations: {ep_dir}")


def episode_cae(ep_dir: Path) -> dict:
    ep = json.loads((ep_dir / "episode.json").read_text())
    graph = json.loads((ep_dir / "task_graph.json").read_text())
    objects = list(dict.fromkeys(g["subject"] for g in graph["success_program"]))
    ref_dir = Path(ep["reference"])
    provenance = verify_binding(ep_dir, ref_dir)
    arts = episode_articulations(graph, ref_dir)
    ref = entity_tracks(ref_dir / "trajectory.json", objects, arts, ref_dir)
    pred = entity_tracks(ep_dir / "native_run/simulation/trajectory.json", objects, arts, ref_dir)
    for label, tracks in (("reference", ref), ("episode", pred)):
        require_entity_tracks(tracks, ["end_effector", *objects], label)
    meta = _capture_metadata(ep_dir, ep)
    done = ((meta.get("completion_evidence") or {}).get("native_predicate_evidence") or {}).get("measured_at_s")
    res = score(ref, pred, pred_end_s=done if ep.get("task_success") else None)
    res.update(source="simulator_execution", episode=ep_dir.name, task_success=ep.get("task_success"),
               reference_provenance=provenance)
    return res


def main() -> None:
    for d in sys.argv[1:]:
        r = episode_cae(Path(d))
        ent = {k: {x: round(v, 3) for x, v in e.items() if x in ("score", "track", "speed", "pace_ratio",
                                                                 "prediction_onset_delay_s")}
               for k, e in r.get("entities", {}).items()}
        print(json.dumps({"episode": r["episode"], "success": r["task_success"],
                          "CAE": None if r["score"] is None else round(r["score"], 3), "entities": ent}))


if __name__ == "__main__":
    main()
