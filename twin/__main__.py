"""Command line: python -m twin <command>

  train                       fit the world model, save it, run the acceptance tests
  validate                    re-run the acceptance tests on the saved world model
  explore PROFILE_ID          sample one person's possible lives and write an HTML report
      --policy rules|neural|foresight   who decides (default: the rule persona)
      --foresight 0.5         knob for --policy foresight: 0 = as they tend to decide, 1 = most foresight
      --paths 2000 --seed 1
      --answers answers.json  questionnaire answers, 1-5 per trait (see twin/examples)
      --plan '{"head": "status", "option": 1, "prob": 0.6, "when": "2026-01"}'   (repeatable)
      --custom person.json    a person who is not in the data (see twin/examples)
      --out report.html --json result.json --no-what-ifs --no-spread --no-foresight
  llm-demos                   ask an LLM (OpenAI; needs OPENAI_API_KEY) to role-play personas at decision points
      --calls 300 --jan-2026-calls 60 --model gpt-4o-mini --workers 8 --dry-run
  replay                      score monthly updates on the held-out 2026 ledger
  clone [--llm] [--llm-sharpen 1,2,2,1]  train the neural persona on the rule persona (plus the LLM answers) and compare
  clone-snapshots             train and evaluate a GRU on January 2026 vLLM labels
  foresight                   train the foresight knob on top of the neural persona and compare
      --points 2000           decision points to look ahead from (each: 11 options x 6 rollouts x 60 months)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config as C


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m twin", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--no-validate", action="store_true")
    sub.add_parser("validate")
    replay = sub.add_parser("replay", help="score monthly updates against the 2026 ledger")
    replay.add_argument("--out", type=Path, default=C.ARTIFACTS / "monthly_replay.json")
    ex = sub.add_parser("explore")
    ex.add_argument("profile_id", nargs="?")
    ex.add_argument("--custom", type=Path)
    ex.add_argument("--paths", type=int, default=2000)
    ex.add_argument("--seed", type=int, default=C.SEED)
    ex.add_argument("--answers", type=Path)
    ex.add_argument("--plan", action="append", default=[])
    ex.add_argument("--temperature", type=float)
    ex.add_argument("--out", type=Path)
    ex.add_argument("--json", type=Path)
    ex.add_argument("--no-what-ifs", action="store_true")
    ex.add_argument("--no-spread", action="store_true")
    ex.add_argument("--no-foresight", action="store_true")
    ex.add_argument("--policy", choices=["rules", "neural", "foresight"], default="rules")
    ex.add_argument("--foresight", type=float, default=0.5)
    ex.add_argument("--markets", choices=["learned", "priors"], default="learned",
                    help="job offers, admission and flats: learned market model or the fixed rates in priors.yaml")
    ld = sub.add_parser("llm-demos")
    ld.add_argument("--calls", type=int, default=300)
    ld.add_argument("--jan-2026-calls", type=int, default=None,
                    help="additional January 2026 ledger snapshots (default: 60, 0 disables)")
    ld.add_argument("--model")
    ld.add_argument("--workers", type=int, default=8)
    ld.add_argument("--dry-run", action="store_true")
    cl = sub.add_parser("clone")
    cl.add_argument("--profiles", type=int, default=400)
    cl.add_argument("--llm", action="store_true", help="use twin/artifacts/llm_demos.json from `llm-demos`")
    cl.add_argument("--llm-weight", type=float, default=10.0)
    cl.add_argument("--llm-sharpen", default="1,2,2,1", help="power on LLM probabilities per head: status,housing,spending,assets (1 = as answered)")
    snapshots = sub.add_parser("clone-snapshots")
    snapshots.add_argument("--labels", type=Path, default=C.ARTIFACTS / "jan_2026_vllm_responses.json")
    fs = sub.add_parser("foresight")
    fs.add_argument("--points", type=int, default=2000)
    a = ap.parse_args(argv)

    if a.cmd == "train":
        from .train import main as train_main
        train_main(run_validation=not a.no_validate)
    elif a.cmd == "replay":
        from .replay import run
        print(json.dumps(run(a.out), indent=2))
    elif a.cmd == "validate":
        from . import validate
        from .train import to_json
        from .world.flow import FlowModel
        res = validate.run(fm_final=FlowModel.load(C.ARTIFACTS / "flow"))
        (C.ARTIFACTS / "validation.json").write_text(json.dumps(to_json(res), indent=1))
        for line in validate.summary_lines(res):
            print(line)
    elif a.cmd == "explore":
        from . import report
        from .explore import default_what_ifs, explore, person_table
        from .policy.traits import Intention, from_profile
        from .train import to_json
        custom = json.loads(a.custom.read_text()) if a.custom else None
        if not (a.profile_id or custom):
            ap.error("give a PROFILE_ID (e.g. P00042) or --custom person.json")
        person = person_table(a.profile_id, custom).iloc[0]
        persona = from_profile(person)
        if a.answers:
            persona = persona.with_answers(json.loads(a.answers.read_text()))
        persona.intentions = [Intention.parse(json.loads(p)) for p in a.plan]
        if a.temperature:
            persona.temperature = a.temperature
        what_ifs = None if a.no_what_ifs else default_what_ifs(person, persona)
        if a.policy == "neural":
            from .policy.net import NeuralPolicy as policy_cls
        elif a.policy == "foresight":
            from .policy.foresight import at_level
            policy_cls = at_level(a.foresight)
        else:
            from .policy.rules import RulePolicy as policy_cls
        from .policy.foresight import PATH as FORESIGHT_NET
        levels = None if (a.no_foresight or not FORESIGHT_NET.exists()) else (0.0, 0.5, 1.0)
        from .world.env import Config, World
        world = World.load(config=Config(markets=a.markets == "learned"))
        res, _ = explore(a.profile_id, custom, persona, a.paths, a.seed, world=world, what_ifs=what_ifs,
                         with_spread=not a.no_spread, policy_cls=policy_cls, foresight_levels=levels)
        res = to_json(res)
        print(f"{res['person']['name']} ({res['persona']['policy']}): high goal {res['high_goal_odds']:.0%}, "
              f"ever below zero {res['shortfall_odds']:.0%}, median net worth 2034 CHF {res['final_net_worth']['p50']:,.0f}")
        for s in res["stories"][:3]:
            print(f"  {s['share']:.0%}  {s['story']}")
        for lv, v in (res.get("foresight") or {}).items():
            print(f"  foresight {lv}: high goal {v['high_goal_odds']:.0%}, ever below zero {v['shortfall_odds']:.0%}, "
                  f"median net worth CHF {v['median_final_net_worth']:,.0f}")
        out = a.out or C.ARTIFACTS / "reports" / f"{a.profile_id or (custom or {}).get('id', 'custom')}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        report.write(res, out)
        print(f"report: {out}")
        if a.json:
            a.json.write_text(json.dumps(res, indent=1))
    elif a.cmd == "llm-demos":
        from .policy.llm import DEFAULT_JAN_2026_CALLS, collect
        jan_calls = DEFAULT_JAN_2026_CALLS if a.jan_2026_calls is None else a.jan_2026_calls
        collect(calls=a.calls, model=a.model, workers=a.workers, dry_run=a.dry_run, jan_2026_calls=jan_calls)
    elif a.cmd == "clone":
        from .policy.net import main as clone_main
        clone_main(n_profiles=a.profiles, use_llm=a.llm, llm_weight=a.llm_weight, llm_sharpen=tuple(float(x) for x in a.llm_sharpen.split(",")))
    elif a.cmd == "clone-snapshots":
        from .policy.net import train_snapshot_proxy
        train_snapshot_proxy(a.labels)
    elif a.cmd == "foresight":
        from .policy.foresight import main as foresight_main
        foresight_main(n_points=a.points)


if __name__ == "__main__":
    main()
