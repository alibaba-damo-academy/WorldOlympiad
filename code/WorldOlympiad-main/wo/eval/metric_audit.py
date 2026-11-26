"""Fine metric summaries; explicit applicability and scorer-specific evidence.

Means give each case/model/seed one vote. Chunk and object component means are
formed within an episode first. No scorer weights or episode scores change.
"""
from math import isfinite
from pathlib import Path


QUESTIONS = {
    "mechanics": ("gravity", "buoyancy", "compression", "impact"),
    "thermotics": ("melting", "sublimation", "vaporization", "condensation", "deposition", "freezing"),
    "material": ("color_mixing", "solubility", "hardness", "combustibility"),
}


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def metric_audit(rows):
    buckets = {}
    probes = sorted({name for row in rows for name in
                     (row.get("physical_probe_details") or {}).get("per_probe", {})}
                    | {p["probe"] for row in rows for p in
                       (row.get("physical_probe_details") or {}).get("probes", [])})

    dimensions = {name: list(ids) for name, ids in QUESTIONS.items()}
    for row in rows:
        for q in (row.get("physical_details") or {}).get("results", []):
            if q.get("dimension") and q.get("question_id"):
                ids = dimensions.setdefault(q["dimension"], [])
                if q["question_id"] not in ids:
                    ids.append(q["question_id"])

    for row in rows:
        episode = row.get("episode", "")
        scorefile = str(Path(episode) / "case/score.json")

        def payload(key):
            result = row.get(key) or {}
            if row.get("status") == "not_applicable" and row.get("reason"):
                return {**result, "status": "not_applicable", "reason": row["reason"]}
            return result

        def add(name, value, payload, source, pointer, *, na=None, pending=None, scale="0-1"):
            explicit_na = payload.get("status") == "not_applicable" and payload.get("reason")
            if na or explicit_na:
                status, value, reason = "not_applicable", None, na or payload["reason"]
            elif pending or payload.get("status") in {"pending", "failed", "error", "unmeasurable"} or not numeric(value):
                status, value, reason = "pending", None, pending or payload.get("reason") or "required measurement missing"
            else:
                status, reason = "scored", None
            observation = {"case_id": row.get("case_id"), "model": row.get("model"),
                           "seed": row.get("seed"), "status": status, "value": value,
                           "source": source, "evidence": pointer}
            if reason:
                observation["reason"] = reason
            b = buckets.setdefault(name, {"scale": scale, "observations": []})
            b["observations"].append(observation)

        p = payload("physical_details")
        psource = "vlm_video_physics"
        add("physical_details.score", p.get("score"), p, psource, scorefile + "#/physical/score")
        questions = {q["question_id"]: q for q in p.get("results", []) if "question_id" in q}
        for dimension, ids in dimensions.items():
            for qid in ids:
                q = questions.get(qid, {})
                na = q.get("relevance_reason") or "scorer explicitly declared unrelated" if q.get("related") is False else None
                complete = q.get("related") is True and isinstance(q.get("compliant"), bool) and (q["compliant"] is False or numeric(q.get("confidence")))
                value = (q["confidence"] if q["compliant"] else 0.) if complete else None
                add(f"physical_details.questions.{qid}", value, p, psource,
                    scorefile + f"#/physical/results/question_id={qid}", na=na)
                buckets[f"physical_details.questions.{qid}"]["observations"][-1].update(
                    related=q.get("related"), compliant=q.get("compliant"), confidence=q.get("confidence"))
            all_unrelated = all(questions.get(qid, {}).get("related") is False for qid in ids)
            unknown = any(qid not in questions or questions[qid].get("related") not in (True, False)
                          or (questions[qid].get("related") is True and (not isinstance(questions[qid].get("compliant"), bool)
                              or (questions[qid].get("compliant") is True and not numeric(questions[qid].get("confidence")))))
                          for qid in ids)
            add(f"physical_details.dimension_scores.{dimension}", (p.get("dimension_scores") or {}).get(dimension),
                p, psource, scorefile + f"#/physical/dimension_scores/{dimension}",
                na="all dimension questions explicitly unrelated" if all_unrelated else None,
                pending="dimension has unresolved questions" if unknown else None)

        p = payload("physical_probe_details")
        source = p.get("source") or "unspecified_task_physics_source"
        evidence = str(Path(episode) / "native_run/simulation/trajectory.json")
        for key in ("score",):
            add("physical_probe_details." + key, p.get(key), p, source, evidence)
        for key in ("plausibility", "task_events"):
            applicability = (p.get("subscore_applicability") or {}).get(key) or p
            add("physical_probe_details.subscores." + key, (p.get("subscores") or {}).get(key), applicability, source, evidence)
        declared = {x["probe"] for x in p.get("probes", [])}
        measured = p.get("per_probe") or {}
        declaration_complete = (p.get("status") == "scored" and isinstance(p.get("probe_count"), int)
                                and p["probe_count"] > 0 and len(p.get("probes", [])) == p["probe_count"]
                                and all(numeric(x.get("score")) or (x.get("status") == "not_applicable" and x.get("reason")) for x in p["probes"]))
        for key in probes:
            records = [x for x in p.get("probes", []) if x["probe"] == key]
            conditional_na = records and all(x.get("status") == "not_applicable" and x.get("reason") for x in records)
            unresolved = any(not numeric(x.get("score")) and not (x.get("status") == "not_applicable" and x.get("reason")) for x in records)
            add("physical_probe_details.per_probe." + key, measured.get(key), p, source, evidence,
                na=("; ".join(sorted({x["reason"] for x in records})) if conditional_na else
                    "outside this episode's fully resolved declared primitive probe set" if key not in declared and declaration_complete else None),
                pending="declared probe measurement missing" if unresolved else None)

        p = payload("geometry_details")
        source = "video_reconstruction:" + str(p.get("reconstruction_model_name", "unspecified")) + ";judge:" + str(p.get("scorer_type", "unspecified"))
        for key in ("gs_score", "meta_score", "camera_motion_score", "final_score_normalized"):
            add("geometry_details." + key, p.get(key), p, source, scorefile + "#/three_d/" + key)

        p = payload("interaction_details")
        source = "vlm_video_interaction:" + str(p.get("backend", "unspecified"))
        add("interaction_details.score", p.get("score"), p, source, scorefile + "#/interaction/score")
        for group, keys in (("chunks", ("visual_quality", "text_alignment", "overall")),
                            ("transitions", ("transition_smoothness", "overall"))):
            entries = p.get(group) or []
            for key in keys:
                values = [x.get(key) for x in entries]
                complete = bool(values) and all(numeric(v) for v in values)
                value = sum(values) / len(values) if complete else None
                na = "single chunk has no chunk boundary" if group == "transitions" and len(p.get("chunks") or []) == 1 and p.get("transitions") == [] else None
                add(f"interaction_details.{group}.{key}", None if value is None else value / 5, p, source,
                    scorefile + f"#/interaction/{group}/*/{key}", na=na, scale="0-1; normalized from 0-5, within-episode mean")
        for key in ("long_range_consistency", "global_text_alignment", "overall"):
            value = (p.get("global") or {}).get(key)
            add("interaction_details.global." + key, value / 5 if numeric(value) else None, p, source,
                scorefile + "#/interaction/global/" + key, scale="0-1; normalized from 0-5")
        summary = p.get("summary") or {}
        components = [summary.get(k) for k in ("chunk_mean", "transition_mean", "global_score")]
        mllm = sum(components) / 15 if all(numeric(x) for x in components) else None
        clip = payload("clip_details")
        raw = (clip.get("summary") or {}).get("semantic_adherence")
        aux = max(0., min(1., (raw - .2) / .2)) if numeric(raw) else None
        clipsource = "clip_frame_caption_similarity:" + str(clip.get("clip_model", "unspecified"))
        add("clip_details.raw", raw, clip, clipsource, scorefile + "#/clip_interaction/summary/semantic_adherence", scale="-1 to 1")
        add("clip_details.aux", aux, clip, clipsource, scorefile + "#/clip_interaction/summary/semantic_adherence")
        paper_interaction = .9 * mllm + .1 * aux if mllm is not None and aux is not None else None
        add("paper_formula.interaction", paper_interaction, p, source + ";" + clipsource,
            scorefile + "#/interaction/summary;#/clip_interaction/summary",
            scale="0-1; paper Eq.9-11 diagnostic, separate from recorded release-code score")

        p = payload("cae_details")
        source = p.get("source") or "unspecified_cae_source"
        evidence = str(Path(episode) / "native_run/simulation/trajectory.json") + ";" + str(Path(episode).parent.parent / "references" / str(row.get("case_id")) / "trajectory.json")
        for key in ("score", "end_effector_score", "object_score"):
            add("cae_details." + key, p.get(key), p, source, evidence)
        entities = p.get("entities") or {}
        for kind in ("end_effector", "object"):
            relevant = [e for e in entities.values() if e.get("kind") == kind]
            for key in ("score", "track", "speed"):
                values = [e.get(key) for e in relevant]
                value = sum(values) / len(values) if values and all(numeric(v) for v in values) else None
                add(f"cae_details.{kind}.{key}", value, p, source, evidence,
                    pending="required entity measurement missing" if not relevant and not (p.get("status") == "not_applicable" and p.get("reason")) else None)
        for name, entity in entities.items():
            for key in ("score", "track", "speed"):
                add(f"cae_details.entities.{name}.{key}", entity.get(key), p, source, evidence)

    def summarize(observations):
        valid = [x for x in observations if x["status"] == "scored"]
        result = {"subset_N": len(valid), "N": len(valid),
                "NA": sum(x["status"] == "not_applicable" for x in observations),
                "pending": sum(x["status"] == "pending" for x in observations),
                "mean": sum(x["value"] for x in valid) / len(valid) if valid else None,
                "subset": [{k: x[k] for k in ("case_id", "model", "seed")} for x in valid],
                "source_evidence": sorted({x["source"] for x in observations})}
        if observations and all("compliant" in x for x in observations):
            compliant = sum(x.get("compliant") is True for x in valid)
            result.update(compliant_N=compliant, binary_pass_rate=compliant / len(valid) if valid else None)
        return result

    result = {}
    for name, bucket in sorted(buckets.items()):
        obs = bucket["observations"]
        result[name] = {**summarize(obs), "scale": bucket["scale"],
                        "aggregation": "equal episode means; object/chunk means computed within episode",
                        "by_model": {model: summarize([x for x in obs if x["model"] == model])
                                     for model in sorted({x["model"] for x in obs})},
                        "observations": obs}
    return result
