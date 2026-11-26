#!/usr/bin/env python3
"""Route and materialize embodiment-matched cases from HuggingFace WorldOlympiad 2.0.

Dataset: https://huggingface.co/datasets/Chengzhiwww/WorldOlympiad2.0
Produces benchmark manifests compatible with `scripts/benchmark.py` (`wov2_benchmark.v1`).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HF_REPO = "Chengzhiwww/WorldOlympiad2.0"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cache" / "hf_worldolympiad2"

# Exact embodiment & domain routing matrix aligned with the WorldOlympiad 2.0 paper
DUAL_ARM_EMBODIMENTS = {"ALOHA", "ALOHA-AgileX", "AgileX Cobot Magic", "ARX5"}
SINGLE_ARM_EMBODIMENTS = {"Franka", "Panda"}
SIM_SOURCES = {"RoboDojo", "ManiSkill3", "Constructed"}
REAL_SOURCES = {"Table30", "RoboCOIN", "RoboMIND"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_hf_file(rel_path: str, cache_dir: Path, *, repo: str = HF_REPO) -> Path:
    dest = cache_dir / rel_path
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{rel_path}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "worldolympiad2-router/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    return dest


def load_cases(cache_dir: Path, *, cases_jsonl: Path | None = None, repo: str = HF_REPO) -> list[dict]:
    path = cases_jsonl if cases_jsonl is not None else download_hf_file("benchmark/cases.jsonl", cache_dir, repo=repo)
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_arm(embodiment: str) -> str:
    if embodiment in DUAL_ARM_EMBODIMENTS:
        return "dual"
    if embodiment in SINGLE_ARM_EMBODIMENTS:
        return "single"
    low = embodiment.lower()
    if any(tok in low for tok in ("aloha", "arx5", "agilex", "cobot", "bimanual", "dual")):
        return "dual"
    return "single"


def normalize_domain(row: dict) -> str:
    dom = str(row.get("domain", "")).lower()
    if dom in ("sim", "simulation"):
        return "sim"
    if dom in ("real", "real_world", "real-world"):
        return "real"
    return "sim" if row.get("source") in SIM_SOURCES else "real"


def recommend_execution_mode(row: dict, execution: str) -> str:
    """Assign preferred execution protocol: use sim_rollout when runnable in sim, else i2av (real/video)."""
    dom = normalize_domain(row)
    if execution == "real_only":
        return "i2av"
    if execution == "sim_only":
        return "sim_rollout"
    # Hybrid sim_and_real (default for all models except sim-only policies like GR00T):
    # prefer native sim_rollout on sim cases when available, fall back to i2av on real/non-sim cases.
    return "sim_rollout" if dom == "sim" else "i2av"


def filter_cases(
    rows: list[dict],
    *,
    arm: str = "all",
    execution: str = "sim_and_real",
    subset: str = "all",
    sources: list[str] | None = None,
    task_families: list[str] | None = None,
    model_split: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Filter WorldOlympiad 2.0 cases by model embodiment and execution capability.

    execution modes:
      - sim_and_real (alias: video_all): hybrid sim + real — prefer sim_rollout where
        simulation is available, fall back to i2av on real-world/video cases (default for
        all embodied world models except sim-only policies like GR00T).
      - sim_only: simulation cases only (for closed-loop sim policies e.g. GR00T N1.6-DROID)
      - real_only: real-world public cases only
      - constructed_only: in-repo constructed canonical MP4 tasks only (direct HF download)
    """
    if execution == "video_all":
        execution = "sim_and_real"
    src_filter = {s.lower() for s in sources} if sources else None
    fam_filter = {f.lower() for f in task_families} if task_families else None
    out = []
    for row in rows:
        row_arm = normalize_arm(str(row.get("embodiment", "")))
        row_dom = normalize_domain(row)
        row_subset = str(row.get("subset", "")).lower()
        row_source = str(row.get("source", ""))
        row_family = str(row.get("task_family", ""))

        if arm != "all" and row_arm != arm:
            continue
        if subset != "all" and row_subset != subset:
            continue
        if execution == "sim_only" and row_dom != "sim":
            continue
        if execution == "real_only" and row_dom != "real":
            continue
        if execution == "constructed_only" and row_subset != "constructed":
            continue
        if src_filter and row_source.lower() not in src_filter:
            continue
        if fam_filter and row_family.lower() not in fam_filter:
            continue
        if model_split and model_split not in (row.get("models") or []):
            continue
        enriched = dict(row)
        enriched["recommended_mode"] = recommend_execution_mode(row, execution)
        out.append(enriched)
        if limit is not None and len(out) >= limit:
            break
    return out


def extract_first_frame(video_path: Path, out_png: Path) -> None:
    """Extract frame 0 only for model conditioning (never leaking future reference frames)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            cv2.imwrite(str(out_png), frame)
            return
    except Exception:
        pass

    try:
        import imageio_ffmpeg  # type: ignore

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("Neither OpenCV nor ffmpeg is available to extract initial conditioning frame.")
    subprocess.run(
        [ffmpeg_bin, "-y", "-i", str(video_path), "-frames:v", "1", str(out_png)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def cmd_summary(args: argparse.Namespace) -> int:
    rows = load_cases(Path(args.cache_dir), cases_jsonl=Path(args.cases_jsonl) if args.cases_jsonl else None, repo=args.repo)
    counts = collections.Counter()
    for r in rows:
        counts[(normalize_arm(str(r.get("embodiment", ""))), normalize_domain(r), r.get("subset"), r.get("source"), r.get("embodiment"))] += 1
    print(f"Dataset: {args.repo} ({len(rows)} benchmark cases + 300 separate real_robot bimanual demos)")
    print(f"{'ARM':<8} {'DOMAIN':<8} {'SUBSET':<12} {'SOURCE':<14} {'EMBODIMENT':<20} {'CASES':>6}")
    print("-" * 74)
    for (arm, dom, subset, src, emb), n in sorted(counts.items()):
        print(f"{arm:<8} {dom:<8} {str(subset):<12} {str(src):<14} {str(emb):<20} {n:>6}")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    rows = load_cases(Path(args.cache_dir), cases_jsonl=Path(args.cases_jsonl) if args.cases_jsonl else None, repo=args.repo)
    selected = filter_cases(
        rows,
        arm=args.arm,
        execution=args.execution,
        subset=args.subset,
        sources=args.source,
        task_families=args.task_family,
        model_split=args.model_split,
        limit=args.limit,
    )
    payload = {
        "schema_version": "wov2_hf_route.v1",
        "source_repo": args.repo,
        "criteria": {
            "arm": args.arm,
            "execution": args.execution,
            "subset": args.subset,
            "sources": args.source or [],
            "task_families": args.task_family or [],
            "model_split": args.model_split,
        },
        "total_matched": len(selected),
        "by_source": dict(collections.Counter(r.get("source") for r in selected)),
        "cases": selected,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {len(selected)} routed cases to {out_path}")
    else:
        print(json.dumps({k: v for k, v in payload.items() if k != "cases"}, indent=2, ensure_ascii=False))
        for r in selected[:10]:
            print(f"  - {r['case_id']} [{r['subset']}/{r['source']}/{r['embodiment']}/{r['task_family']}]")
        if len(selected) > 10:
            print(f"  ... ({len(selected) - 10} more)")
    return 0


def cmd_materialize(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_cases(cache_dir, cases_jsonl=Path(args.cases_jsonl) if args.cases_jsonl else None, repo=args.repo)
    selected = filter_cases(
        rows,
        arm=args.arm,
        execution=args.execution,
        subset=args.subset,
        sources=args.source,
        task_families=args.task_family,
        model_split=args.model_split,
        limit=args.limit,
    )
    if not selected:
        raise SystemExit("No cases matched the requested routing filters.")

    manifest_cases = []
    skipped_upstream = []
    for row in selected:
        cid = row["case_id"]
        ref_rel = row.get("reference_video")
        if not ref_rel:
            skipped_upstream.append({
                "case_id": cid,
                "source": row.get("source"),
                "upstream_repo": row.get("upstream_repo"),
                "upstream_revision": row.get("upstream_revision"),
                "upstream_media": row.get("upstream_media"),
            })
            continue
        cached_mp4 = download_hf_file(ref_rel, cache_dir, repo=args.repo)
        expected_sha = row.get("expected_reference_sha256")
        if expected_sha and sha256_file(cached_mp4) != expected_sha:
            raise SystemExit(f"SHA-256 mismatch for {cid} ({ref_rel})")

        case_dir = out_dir / "cases" / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        local_mp4 = case_dir / "reference.mp4"
        if not local_mp4.exists():
            shutil.copy2(cached_mp4, local_mp4)
        init_png = case_dir / "initial_front.png"
        if not init_png.exists():
            extract_first_frame(local_mp4, init_png)

        manifest_cases.append({
            "case_id": cid,
            "domain": normalize_domain(row),
            "instruction": row["instruction"],
            "reference_view": "front",
            "reference_video": f"cases/{cid}/reference.mp4",
            "conditioning": {
                "front": f"cases/{cid}/initial_front.png",
            },
            "metadata": {
                "subset": row.get("subset"),
                "source": row.get("source"),
                "embodiment": row.get("embodiment"),
                "arm": normalize_arm(str(row.get("embodiment", ""))),
                "task_family": row.get("task_family"),
                "recommended_mode": row.get("recommended_mode", recommend_execution_mode(row, args.execution)),
            },
        })

    if not manifest_cases:
        raise SystemExit(
            f"All {len(skipped_upstream)} selected cases require upstream public dataset media "
            f"(not in-repo MP4s). Use --execution constructed_only for direct HF MP4 materialization, "
            f"or materialize upstream sources listed in upstream_pending.json."
        )

    dataset_id = f"worldolympiad2_{args.arm}_{args.execution}"
    dataset_doc = {
        "schema_version": "wov2_benchmark.v1",
        "dataset_id": dataset_id,
        "version": "2026.11.25",
        "source_repo": args.repo,
        "cases": manifest_cases,
    }
    manifest_path = out_dir / "dataset.json"
    manifest_path.write_text(json.dumps(dataset_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if skipped_upstream:
        (out_dir / "upstream_pending.json").write_text(
            json.dumps(skipped_upstream, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"PASS: materialized {len(manifest_cases)} verified cases -> {manifest_path}")
    if skipped_upstream:
        print(f"NOTE: {len(skipped_upstream)} public upstream-indexed cases saved to {out_dir / 'upstream_pending.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=HF_REPO, help=f"HuggingFace dataset repo (default: {HF_REPO})")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help="Local cache directory")
    p.add_argument("--cases-jsonl", default=None, help="Optional local cases.jsonl for offline routing")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary", help="Print embodiment/domain/source routing table")

    for name in ("route", "materialize"):
        sp = sub.add_parser(name, help=f"{name.capitalize()} embodiment-matched cases")
        sp.add_argument("--arm", choices=["all", "dual", "single"], default="all")
        sp.add_argument(
            "--execution",
            choices=["sim_and_real", "sim_only", "real_only", "constructed_only", "video_all"],
            default="sim_and_real",
            help="sim_and_real (default: sim_rollout where available, i2av on real/video), sim_only (e.g. GR00T), real_only, constructed_only",
        )
        sp.add_argument("--subset", choices=["all", "public", "constructed"], default="all")
        sp.add_argument("--source", action="append", help="Filter by source (e.g. RoboDojo, ManiSkill3, Table30, Constructed)")
        sp.add_argument("--task-family", action="append", help="Filter by task family (e.g. cup_hanging, stack)")
        sp.add_argument("--model-split", default=None, help="Filter to a published model split (fastwam, lingbot_va, longwam_cod, cosmos3, gr00t_n1.6_droid)")
        sp.add_argument("--limit", type=int, default=None, help="Maximum number of cases")
        if name == "route":
            sp.add_argument("--out", default=None, help="Optional JSON output path")
        else:
            sp.add_argument("--out-dir", required=True, help="Output directory for dataset.json and cases/")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "summary":
        return cmd_summary(args)
    if args.cmd == "route":
        return cmd_route(args)
    if args.cmd == "materialize":
        return cmd_materialize(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
