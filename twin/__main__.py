"""Command line: python -m twin <command>

  train                       fit the world model, save it, run the acceptance tests
  validate                    re-run the acceptance tests on the saved world model
  explore PROFILE_ID          sample one person's possible lives and write an HTML report
      --policy rules|neural   the rule persona (default) or its neural clone
      --paths 2000 --seed 1
      --answers answers.json  questionnaire answers, 1-5 per trait (see twin/examples)
      --plan '{"head": "status", "option": 1, "prob": 0.6, "when": "2026-09"}'   (repeatable)
      --custom person.json    a person who is not in the data (see twin/examples)
      --out report.html --json result.json --no-what-ifs --no-spread
  clone                       stretch: train the neural persona on the rule persona and compare
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
    ex.add_argument("--policy", choices=["rules", "neural"], default="rules",
                    help="rule persona, or the neural persona cloned from it (run `clone` first)")
    cl = sub.add_parser("clone")
    cl.add_argument("--profiles", type=int, default=400)
    a = ap.parse_args(argv)

    if a.cmd == "train":
        from .train import main as train_main
        train_main(run_validation=not a.no_validate)
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
        else:
            from .policy.rules import RulePolicy as policy_cls
        res, _ = explore(a.profile_id, custom, persona, a.paths, a.seed, what_ifs=what_ifs,
                         with_spread=not a.no_spread, policy_cls=policy_cls)
        res = to_json(res)
        print(f"{res['person']['name']}: high goal {res['high_goal_odds']:.0%}, ever below zero {res['shortfall_odds']:.0%}, "
              f"median net worth 2034 CHF {res['final_net_worth']['p50']:,.0f}")
        for s in res["stories"][:3]:
            print(f"  {s['share']:.0%}  {s['story']}")
        out = a.out or C.ARTIFACTS / "reports" / f"{a.profile_id or (custom or {}).get('id', 'custom')}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        report.write(res, out)
        print(f"report: {out}")
        if a.json:
            a.json.write_text(json.dumps(res, indent=1))
    elif a.cmd == "clone":
        from .policy.net import main as clone_main
        clone_main(n_profiles=a.profiles)


if __name__ == "__main__":
    main()
