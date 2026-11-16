"""Combine three-track judge scores and CAE v2 into one result per episode.

    python -m wo.eval.report --runs RUN_ROOT --out RESULTS.json

RUN_ROOT/<case>/<model>_s<seed>/ must contain episode.json (wo.models.*) and
optionally case/score.json (wo.eval.case --score).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import compose
from .metric_audit import QUESTIONS, metric_audit
from .cae_episodes import episode_cae
from .physical_probes import episode_probes


def _row(value):
    return None if value is None else {"status": "scored", "score": float(value)}


def applicability_result(ep: dict) -> dict:
    """Missing native task measurements remain implementation work, never NA."""
    termination = ep.get("termination_reason") or ""
    declared = termination == "not_applicable" or termination.startswith("not_applicable:")
    if ep.get("status") != "not_applicable" and not declared:
        return {}
    reason = (ep.get("reason") or ep.get("not_applicable_reason") or
              (termination.partition(":")[2].strip() if declared else ""))
    if not reason:
        return {"status": "pending", "reason": "not_applicable declaration lacks reason",
                "pending_tracks": ["applicability_evidence"], "tracks": {},
                "task_success": None, "worldolympiad2_score": None}
    kind = (ep.get("applicability") or {}).get("kind")
    gap = (kind in {"implementation_gap", "missing_measurement", "missing_adapter"}
           or "no measurement" in reason.lower()
           or "cannot represent primitive" in reason.lower()
           # A rigid importer rejecting joints is an adapter gap, not a
           # limitation of the native Isaac simulator or DROID protocol.
           or ("importer" in reason.lower() and
               any(token in reason.lower() for token in ("multi-link", "joints", "articulat")))
           # Replacing a required movable scene host with a fixed table is a
           # scene adapter omission, not evidence that the policy is inapplicable.
           or ("articulated support host" in reason.lower() and
               "replaces the support" in reason.lower())
           or "reference scene is not at rest" in reason.lower())
    # Bare exception text is not an applicability decision. New, unknown
    # importer failures must not silently become exemptions from the matrix.
    intrinsic = kind in {"simulation_representation", "action_protocol", "embodiment_protocol"}
    review = not gap and not intrinsic
    return {"status": "pending" if gap or review else "not_applicable", "reason": reason,
            "applicability_kind": "implementation_gap" if gap else ("applicability_review" if review else kind),
            "pending_tracks": (["native_task_measurement"] if gap else
                               ["applicability_evidence"] if review else []), "tracks": {},
            "task_success": None, "worldolympiad2_score": None}


def physical_result(generic: dict, task: dict) -> dict:
    """Keep both required physical measurements; missing is not an exclusion."""
    sub = {}
    for name, row in (("generic_video", generic), ("task_probes", task)):
        value = row.get("score")
        if row.get("status") == "not_applicable" and row.get("reason"):
            sub[name] = {"status": "not_applicable", "score": None, "reason": row["reason"]}
        elif (not isinstance(value, (int, float)) or
              row.get("status") in {"pending", "failed", "error", "unmeasurable", "not_applicable"}):
            sub[name] = {"status": "pending", "score": None,
                         "reason": row.get("reason") or "required measurement missing"}
        else:
            if not 0 <= float(value) <= 1:
                raise ValueError("physical subscore must lie in [0, 1]")
            sub[name] = {"status": "scored", "score": float(value)}
        if row.get("source"):
            sub[name]["source"] = row["source"]
    live = [r for r in sub.values() if r["status"] != "not_applicable"]
    if not live:
        status, score = "not_applicable", None
    elif any(r["status"] != "scored" for r in live):
        status, score = "pending", None
    else:
        status, score = "scored", sum(r["score"] for r in live) / len(live)
    return {"status": status, "score": score, "submetrics": sub,
            "declared_weights": {name: .5 for name in sub},
            "weights": {name: (1 / len(live) if r["status"] != "not_applicable" else 0.)
                        for name, r in sub.items()}}


def _episode_case_id(ep: dict, ep_dir: Path) -> str:
    """Resolve case identity from the bound reference for native receipts that omit it."""
    case_id = ep.get("case_id")
    if case_id:
        return case_id
    reference = ep.get("reference")
    if reference:
        certificate = Path(reference) / "certificate.json"
        if certificate.is_file():
            bound_case = json.loads(certificate.read_text()).get("case_id")
            if bound_case:
                return bound_case
    # Older adapters may not serialize a case ID, but stage output paths retain
    # the stable <run>/<case>/<model>_s<seed> layout.
    if ep_dir.parent.name and ep_dir.parent.name != ep_dir.name:
        return ep_dir.parent.name
    raise ValueError(f"episode has no case_id and no case-bound reference: {ep_dir}")


def _episode_provenance(ep: dict, ep_dir: Path, case_id: str) -> dict:
    """Expose the immutable task/reference/episode/score chain in each report row."""
    reference = ep.get("reference")
    reference_dir = Path(reference) if isinstance(reference, str) and reference else None
    certificate_path = reference_dir / "certificate.json" if reference_dir else None
    certificate = {}
    if certificate_path and certificate_path.is_file():
        try:
            certificate = json.loads(certificate_path.read_text())
        except (OSError, json.JSONDecodeError):
            certificate = {}

    episode_binding_path = ep_dir / "episode_reference_binding.json"
    episode_binding = {}
    if episode_binding_path.is_file():
        try:
            episode_binding = json.loads(episode_binding_path.read_text())
        except (OSError, json.JSONDecodeError):
            episode_binding = {}
    bound_reference = episode_binding.get("reference_binding") or {}
    task_path = bound_reference.get("task") or certificate.get("task")
    if not task_path:
        candidate = ep_dir.parents[1] / "tasks" / f"{case_id}.json"
        task_path = str(candidate) if candidate.is_file() else None
    task_digest = (bound_reference.get("task_sha256") or certificate.get("task_sha256"))

    score_path = ep_dir / "case" / "score.json"
    score_receipt_path = ep_dir / "case" / "three_track_receipt.json"
    score_receipt = {}
    if score_receipt_path.is_file():
        try:
            score_receipt = json.loads(score_receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            score_receipt = {}

    path = lambda value: str(Path(value).resolve()) if value else None
    return {
        "task": {"case_id": case_id, "path": path(task_path), "sha256": task_digest},
        "reference": {"path": path(reference), "certificate": path(certificate_path),
                      "task_sha256": certificate.get("task_sha256")},
        "episode": {"path": str(ep_dir.resolve()),
                    "metadata": str((ep_dir / "episode.json").resolve()),
                    "reference_binding": path(episode_binding_path)},
        "score": {"path": path(score_path) if score_path.is_file() else None,
                  "receipt": path(score_receipt_path) if score_receipt_path.is_file() else None,
                  "sha256": score_receipt.get("score_sha256")},
    }


def episode_result(ep_dir: Path) -> dict:
    ep = json.loads((ep_dir / "episode.json").read_text())
    case_id = _episode_case_id(ep, ep_dir)
    provenance = _episode_provenance(ep, ep_dir, case_id)
    invalidation_path = ep_dir / "evaluation_invalidated.json"
    if invalidation_path.is_file():
        finding = json.loads(invalidation_path.read_text())
        # Keep raw attempts/scores intact. An explicitly invalidated execution
        # cannot enter current score or success aggregates; a native retry archives
        # the whole attempt (including this finding) through episode_stage.
        return {"episode": str(ep_dir), "case_id": case_id, "model": ep["model"],
                "family": ep.get("task_family", "unknown"), "seed": ep.get("seed"),
                "provenance": provenance,
                "task_success": None, "historical_task_success": ep.get("task_success"),
                "status": "pending", "applicability_kind": "implementation_gap",
                "reason": finding["reason"], "pending_tracks": ["native_adapter_validation"],
                "tracks": {}, "evaluation_invalidation": finding,
                "source": str(invalidation_path),
                "historical_score_source": str(ep_dir / "case/score.json"),
                **{key: None for key in ("physical", "physical_generic", "physical_task",
                   "geometry", "interaction", "semantic_interaction", "cae_v2", "worldolympiad2_score")}}
    score_path = ep_dir / "case" / "score.json"
    s = json.loads(score_path.read_text()) if score_path.is_file() else {}
    p, g, i = s.get("physical") or {}, s.get("three_d") or {}, s.get("interaction") or {}
    family = ep.get("task_family")
    try:
        cert = Path(ep["reference"]) / "certificate.json"
        if not family and cert.is_file():
            family = json.loads(cert.read_text()).get("family")
    except (KeyError, OSError, json.JSONDecodeError):
        pass
    # Physical track = generic video physics (the question bank, VLM judged) and
    # task physics (probes derived from this task's primitives, geometry measured),
    # weighed equally.  The bank alone saturates: a tabletop manipulation makes
    # exactly one of its fourteen questions relevant.
    generic_physical, geometry, semantic = p.get("score"), g.get("final_score_normalized"), i.get("score")
    try:
        probe_row = episode_probes(ep_dir)
    except Exception as exc:  # recorded, not hidden
        probe_row = {"status": "pending", "score": None, "reason": str(exc)}
    task_physical = probe_row.get("score")
    physical_track = physical_result(p, probe_row)
    physical = physical_track["score"]
    try:
        cae = episode_cae(ep_dir)
        # Keep entity-level EE/object track and speed metrics, source and any
        # reconstruction calibration: the scalar alone is not an audit trail.
        cae_row = dict(cae)
    except Exception as exc:  # recorded, not hidden
        cae_row = {"status": "pending", "score": None, "reason": str(exc)}
    combined = compose(physical_track, _row(geometry), _row(semantic), cae_row)
    # Preserve the scorer's full per-question/component payload.  Consumers can
    # therefore inspect confidence, dimension components, and interaction details
    # without depending on scorer-specific field names.
    return {"episode": str(ep_dir), "case_id": case_id, "model": ep["model"],
            "family": family or s.get("task_family") or "unknown", "seed": ep.get("seed"),
            "provenance": provenance,
            "task_success": ep.get("task_success"), "physical": physical,
            "physical_generic": generic_physical, "physical_task": task_physical,
            "physical_task_source": probe_row.get("source"), "geometry": geometry,
            "semantic_interaction": semantic, "cae_v2": cae_row.get("score"),
            "interaction": combined["tracks"]["interaction"]["score"],
            "worldolympiad2_score": combined["worldolympiad2_score"], "status": combined["status"],
            "pending_tracks": combined["pending_tracks"], "tracks": combined["tracks"],
            "physical_details": p, "physical_probe_details": probe_row, "geometry_details": g,
            "interaction_details": i, "clip_details": s.get("clip_interaction") or {}, "cae_details": cae_row,
            **applicability_result(ep)}


def _mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def aggregates(rows):
    def grouped(key):
        out = {}
        for r in rows:
            out.setdefault(r.get(key) or "unknown", []).append(r)
        return {name: {"episodes": len(xs),
                        "not_applicable": sum(x.get('status') == 'not_applicable' for x in xs),
                        "task_success_rate":
                        _mean([{ "success": float(x["task_success"]) } for x in xs
                               if isinstance(x.get("task_success"), bool)], "success"),
                        "success_measured": sum(isinstance(x.get("task_success"), bool) for x in xs),
                        "physical": _mean(xs, "physical"),
                        "physical_generic": _mean(xs, "physical_generic"),
                        "physical_task": _mean(xs, "physical_task"),
                        "geometry": _mean(xs, "geometry"),
                        "interaction": _mean(xs, "interaction"), "total": _mean(xs, "worldolympiad2_score")}
                for name, xs in sorted(out.items())}
    return {"by_model": grouped("model"), "by_family": grouped("family")}


EXPECTED_MODELS = ('cosmos3', 'lingbot_va', 'gr00t', 'fastwam')
MODEL_ALIASES = {'cosmos3_nano_droid': 'cosmos3', 'gr00t_n1.6_droid': 'gr00t'}


def run_manifest(run: Path) -> dict:
    """What matrix a run declares it is producing.

    A run directory accumulates episodes from earlier designs (this benchmark
    once demanded two seeds per case).  The declared matrix, when present, is
    what the audit checks - otherwise stale artifacts silently redefine the
    benchmark being measured.
    """
    path = run / "manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def select_declared(rows: list[dict], seeds) -> tuple[list[dict], list[dict]]:
    """Split episodes into those inside the declared seeds and those outside.

    Episodes left on disk by an earlier design (two seeds per case) must not reach
    the result table or the aggregates: a table that mixes designs is the thing a
    declared matrix exists to prevent.  Without declared seeds nothing is excluded.
    """
    if not seeds:
        return rows, []
    keep = {int(x) for x in seeds}
    return ([r for r in rows if r.get("seed") in keep], [r for r in rows if r.get("seed") not in keep])


def requested_seeds(rows: list[dict], explicit=None) -> list[int]:
    """Seeds the run actually uses.

    The benchmark used to demand two seeds; it now runs one per case.  The set is
    taken from the episodes that exist (and from --seeds when given) so a run is
    never audited against a matrix it was not asked to produce.
    """
    if explicit is not None:
        return sorted({int(x) for x in explicit})
    found = {int(r["seed"]) for r in rows if isinstance(r.get("seed"), int)}
    return sorted(found) if found else [0]


def select_declared_cases(rows: list[dict], cases) -> tuple[list[dict], list[dict]]:
    """Keep only episodes in an explicit run case scope; absent scope keeps all."""
    if cases is None:
        return rows, []
    keep = set(cases)
    return ([row for row in rows if row.get('case_id') in keep],
            [row for row in rows if row.get('case_id') not in keep])


def coverage(run: Path, rows: list[dict], models=None, seeds=None) -> dict:
    """Audit the requested matrix without inventing results for absent adapters.

    The matrix is the manifest's when the run declares one, else the caller's
    arguments, else what the episodes show.  Missing episodes remain pending, not
    failed or not_applicable.  Only measured episodes enter score aggregates.
    References must have an explicit success.
    """
    from ..agent.episode_stage import declared_task_paths

    task_paths = declared_task_paths(run)
    tasks = [json.loads(path.read_text()) for path in task_paths]
    manifest = run_manifest(run)
    models = tuple(models or manifest.get("models") or EXPECTED_MODELS)
    declared = seeds if seeds is not None else manifest.get("seeds")
    seeds = tuple(requested_seeds(rows, declared))
    measured = {(r['case_id'], MODEL_ALIASES.get(r['model'], r['model']), r.get('seed')): r
                for r in rows}
    from ..agent.episode_stage import latest_reference_jobs
    from .reference_binding import verify_task

    jobs = latest_reference_jobs(run)
    families, missing, references = {}, [], []
    for task in tasks:
        case, family = task['case_id'], task['family']
        families.setdefault(family, []).append(case)
        ref = run / 'references' / case
        cert_path = ref / 'certificate.json'
        try:
            verify_task(ref, run / 'tasks' / f'{case}.json')
            ready = (ref / 'replay/exterior_left.mp4').is_file() and jobs.get(case) == 'done'
        except (OSError, ValueError, KeyError):
            ready = False
        if not ready:
            references.append({'case_id': case, 'family': family,
                               'status': 'failed' if jobs.get(case) == 'failed' or
                               (cert_path.is_file() and jobs.get(case, 'done') == 'done') else 'pending'})
        for model in models:
            for seed in seeds:
                row = measured.get((case, model, seed))
                if row is None or row.get('status') not in {'scored', 'complete', 'not_applicable'}:
                    missing.append({'case_id': case, 'family': family, 'model': model,
                                    'seed': seed, 'status': 'pending',
                                    'reason': 'episode missing' if row is None else 'scoring incomplete'})
                elif row.get('status') == 'not_applicable' and not row.get('reason'):
                    missing.append({'case_id': case, 'family': family, 'model': model,
                                    'seed': seed, 'status': 'pending',
                                    'reason': 'not_applicable lacks reason'})
    return {'models': list(models), 'seeds': list(seeds), 'cases': len(tasks),
            'expected_episodes': len(tasks) * len(models) * len(seeds),
            'family_cases': families, 'missing_references': references,
            'pending_matrix_entries': missing,
            'matrix_complete': bool(tasks) and not references and not missing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="seeds the run must cover; default: infer from the episodes")
    a = ap.parse_args()
    rows = [episode_result(p.parent) for p in sorted(a.runs.glob("*/*/episode.json"))]
    manifest = run_manifest(a.runs)
    declared = tuple(a.seeds) if a.seeds else tuple(manifest.get("seeds") or ())
    rows, excluded = select_declared(rows, declared)
    from ..agent.episode_stage import declared_task_paths
    declared_cases = ([path.stem for path in declared_task_paths(a.runs)]
                      if "cases" in manifest else None)
    rows, excluded_cases = select_declared_cases(rows, declared_cases)
    document = {"schema_version": "wo_results.v2", "episodes": rows, **aggregates(rows),
                "coverage": coverage(a.runs, rows, seeds=declared or None),
                "metric_audit": metric_audit(rows),
                "protocol": {"model_time_limit_seconds_by_default": {"fastwam": None, "longwam": None, "cosmos3": 60, "gr00t": 60, "lingbot_va": 60},
                             "native_episode_window": "FastWAM and Long-WAM default to stable native task success or the native action limit; explicit fixed-window and historical rows retain their recorded termination",
                             "legacy_fastwam_window_rows": sum(1 for row in rows if row.get("model") == "fastwam" and row.get("termination_reason") == "evaluation_window_complete"),
                             "cae_v2_alignment": "active-segment progress alignment; waiting/pace not penalized; WAM simulator trajectories or world-model video reconstruction"},
                "excluded_by_manifest": {"count": len(excluded) + len(excluded_cases),
                                         "seeds": sorted({r.get("seed") for r in excluded if r.get("seed") is not None}),
                                         "cases": sorted({r.get("case_id") for r in excluded_cases if r.get("case_id") is not None})}}
    document["protocol"].update(
        semantic_interaction="recorded release-code score; paper Eq.9-11 is a separate diagnostic",
        clip_aux_calibration={"tau_min": .20, "tau_max": .40, "paper_lambda": .1},
        judge_chunking="32-frame native video intervals with the same task caption; differs from the paper's event-caption annotations",
        paper_reference="https://arxiv.org/html/2606.11129v2",
        cohort="declared native task matrix; not the original 1000-video dataset or an identified original same-scene subset")
    historical = a.runs / "historical_source_audit.json"
    if historical.is_file():
        document["historical_source_audit"] = json.loads(historical.read_text())
    a.out.write_text(json.dumps(document, indent=1))
    md = a.out.with_suffix(".md")
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    lines = ["# WorldOlympiad results", "",
             "Physical = mean(generic video physics, task physics probes); "
             "Interaction = mean(semantic interaction, CAE v2). Sub-scores are listed per episode.",
             "",
             "| case | model | seed | success | physical | phys:generic | phys:task | geometry | semantic | CAE v2 | total |",
             "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case_id']} | {r['model']} | {r['seed']} | "
                     f"{'-' if r['task_success'] is None else ('Y' if r['task_success'] else 'N')} | "
                     f"{fmt(r['physical'])} | {fmt(r.get('physical_generic'))} | {fmt(r.get('physical_task'))} | "
                     f"{fmt(r['geometry'])} | "
                     f"{fmt(r['semantic_interaction'])} | {fmt(r['cae_v2'])} | {fmt(r['worldolympiad2_score'])} |")
    paper_groups = [
        ("Physical subsets (paper Tables 7-8; confidence-weighted release-code scores)",
         [(d, "physical_details.dimension_scores." + d) for d in ("mechanics", "thermotics", "material")]),
        ("Physical rules (paper Table 8; JSON also reports binary pass rates)",
         [(q, "physical_details.questions." + q) for ids in QUESTIONS.values() for q in ids]),
        ("Geometry (paper Tables 10-11)",
         [(k, "geometry_details."+k) for k in ("gs_score", "meta_score", "camera_motion_score", "final_score_normalized")]),
        ("Interaction (paper Tables 9/12; historical scores retained)",
         [("Chunk", "interaction_details.chunks.overall"), ("Trans", "interaction_details.transitions.overall"),
          ("Global", "interaction_details.global.overall"), ("Long Range", "interaction_details.global.long_range_consistency"),
          ("Global Text", "interaction_details.global.global_text_alignment"), ("CLIP Raw", "clip_details.raw"),
          ("CLIP Aux", "clip_details.aux"), ("Paper Eq.9-11", "paper_formula.interaction"),
          ("Release I", "interaction_details.score"), ("CAE", "cae_details.score")])]
    lines += ["", "Paper metric layout on our declared cohort: each cell gives mean (N), NA, pending. The paper names its appendix cohort same-scene but does not publish its case IDs/effective N there. Our native task matrix and chunking are explicitly identified instead."]
    for title, columns in paper_groups:
        lines += ["", "## " + title, "", "| Model | " + " | ".join(label for label, _ in columns) + " |",
                  "|---|" + "---|" * len(columns)]
        for model in sorted(document["by_model"]):
            cells = []
            for _, key in columns:
                entry = document["metric_audit"].get(key, {}).get("by_model", {}).get(model, {})
                cells.append(f"{fmt(entry.get('mean'))} (N={entry.get('N', 0)}); NA={entry.get('NA', 0)}, pending={entry.get('pending', 0)}")
            lines.append("| " + model + " | " + " | ".join(cells) + " |")
    if document.get("historical_source_audit"):
        source_audit = document["historical_source_audit"]
        lines += ["", "## Historical source evidence", "",
                  f"{source_audit['producer_bound_count']} episodes have producer-time binding receipts; {source_audit['retrospectively_checked_count']} historical Cosmos episodes were checked retrospectively against preserved native inputs. Pending audits: {source_audit['pending_count']}. No producer receipt was backfilled. Cabinet's original certificate hash differs; the actual initial state, prompt/joint goal, imported body, compiled mesh hashes and scoring reference video match. See JSON for checks, hashes and limitations."]
    audit = document['coverage']
    gone = document['excluded_by_manifest']
    if gone['count']:
        lines += ['', f"Not shown: {gone['count']} episode(s) from seed(s) {gone['seeds']} on disk are outside "
                      f"the seeds this run declares in manifest.json."]
    lines += ['', '## Fine metric availability and provenance', '',
              'Means weight each episode equally; chunks/objects are averaged within an episode first. N counts measured applicable episodes. Explicit unrelated physics questions and probes outside the fully measured declared primitive set are NA; missing measurement is pending. JSON contains actual subsets, per-model summaries and individual evidence paths. Interaction component tables normalize the recorded 0-5 values to 0-1, as in the paper; CLIP Raw retains -1 to 1.', '',
              '| metric | mean | N | NA | pending | source |',
              '|---|---:|---:|---:|---:|---|']
    for name, metric in document['metric_audit'].items():
        lines.append(f"| {name} | {fmt(metric['mean'])} | {metric['N']} | {metric['NA']} | {metric['pending']} | {'; '.join(metric['source_evidence'])} |")
    lines += ['', '## Evaluation protocol', '',
              'Native episode budgets follow their recorded protocols. FastWAM and Long-WAM default to stable native success or the native action limit; explicit fixed-window and historical rows retain their actual cutoff and overshoot. CAE v2 uses active-segment progress alignment. FastWAM and this Long-WAM IDM control adapter export native execution rather than a separately decoded imagination video. GR00T here is N1.6-DROID, not the user-requested original N1 (arXiv:2503.14734).', '',
              '## Coverage audit', '',
              f"Requested matrix: {audit['cases']} cases x {len(audit['models'])} models x {len(audit['seeds'])} seeds = {audit['expected_episodes']} episodes.",
              f"Matrix complete: {audit['matrix_complete']}. Missing successful references: {len(audit['missing_references'])}. Pending matrix entries: {len(audit['pending_matrix_entries'])}.",
              '', 'Missing episodes are pending, not model failures or not_applicable.', '',
              '```json', json.dumps(audit, indent=2), '```']
    lines += ["", "## Aggregates", "", "```json", json.dumps({"by_model": document["by_model"], "by_family": document["by_family"]}, indent=2), "```"]
    lines += ["", "## Per-episode track details", ""]
    for r in rows:
        lines += [f"### {r['case_id']} / {r['model']} / seed {r['seed']}", "",
                  "```json", json.dumps({k: r[k] for k in
                      ("status", "pending_tracks", "tracks", "physical_details",
                       "physical_probe_details", "geometry_details",
                       "interaction_details", "clip_details", "cae_details")}
                      | ({'reason': r['reason']} if r.get('reason') else {}), indent=2), "```", ""]
    md.write_text("\n".join(lines) + "\n")
    from .report_v3 import write_report
    write_report(document, a.out.with_name(a.out.stem + "_v3" + a.out.suffix))
    print("\n".join(lines[:3]))


if __name__ == "__main__":
    main()
