"""Queue fresh native evidence for explicit adapter invalidations; keep history."""
import argparse
import json
from pathlib import Path
from wo.agent import jobs
from wo.agent.stages import run_dir


def queue_invalidated(run, *, priority=10):
    con=jobs.connect()
    active=[dict(row) for row in con.execute("select * from jobs where status in ('queued','running')")]
    records=[]
    for marker in sorted(run_dir(run).glob("*/*_s*/evaluation_invalidated.json")):
        case=marker.parents[1].name
        model, seed=marker.parent.name.rsplit("_s",1)
        args={"run":run,"case":case,"model":model,"seed":int(seed)}
        def matching(row):
            a=json.loads(row["args"])
            return all(a.get(k,0 if k=="seed" else None)==v for k,v in args.items())
        existing=[row for row in active if matching(row)]
        episodes=[row for row in existing if row["kind"]=="episode"]
        if episodes:
            eid=max(row["id"] for row in episodes)
        else:
            # An active scorer on the old attempt must finish before it is archived.
            deps=[row["id"] for row in existing if row["kind"]=="score" and row["status"]=="running"]
            eid=jobs.submit("episode",{**args,"window_s":60},after=deps,
                            priority=priority,submitted_by="invalidated-attempt-remeasurement")
        scores=[row for row in existing if row["kind"]=="score" and eid in json.loads(row["after"])]
        sid=max(row["id"] for row in scores) if scores else jobs.submit("score",args,after=[eid],
                            priority=priority,submitted_by="invalidated-attempt-remeasurement")
        records.append({**args,"episode":eid,"score":sid,"episode_reused":bool(episodes),"score_reused":bool(scores)})
    return records


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",required=True)
    ap.add_argument("--priority",type=int,default=10)
    a=ap.parse_args()
    print(json.dumps(queue_invalidated(a.run,priority=a.priority),indent=2))


if __name__=="__main__":main()
