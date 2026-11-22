"""Plan versions on disk, one JSON file per person: twin/artifacts/plans/<id>.json (git-ignored).

Each version keeps what the person said (wishes or a nudge), the plan, its simulation and
what changed, so the next nudge starts from the latest plan and knows the earlier updates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import config as C
from .schema import LifePlan

PLANS = C.ARTIFACTS / "plans"


def path_for(person_id: str, root: Path = PLANS) -> Path:
    safe = "".join(ch for ch in person_id if ch.isalnum() or ch in "-_") or "custom"
    return root / f"{safe}.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"versions": []}


def latest(path: Path) -> tuple[LifePlan, dict]:
    record = load(path)
    if not record["versions"]:
        raise FileNotFoundError(f"No plan saved at {path}; draft one first with `python -m twin plan ... --wish`.")
    entry = record["versions"][-1]
    plan, errors = LifePlan.from_dict(entry["plan"])
    if errors:
        raise ValueError(f"The saved plan in {path} is invalid: {errors}")
    return plan, entry


def history(path: Path) -> list[str]:
    """Earlier updates in words, oldest first, for the next nudge's prompt."""
    out = []
    for v in load(path)["versions"]:
        said = v.get("said", "").strip().replace("\n", " ")
        if v["kind"] == "nudge":
            out.append(f"{v.get('as_of') or 'update'}: \"{said[:200]}\" -> {v.get('changes', '')[:200]}")
        elif said:
            out.append(f"first plan, from: \"{said[:200]}\"")
    return out


def append(path: Path, person_id: str, kind: str, said: str, result, as_of=None) -> dict:
    record = load(path)
    entry = {"version": len(record["versions"]) + 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind,
             "said": said, "as_of": as_of, **result.to_dict()}
    record["person"] = person_id
    record["versions"].append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return entry
