"""Fit the world model from the data and check it against the baseline.

    python -m twin train            # fit, save to twin/artifacts, run the acceptance tests
    python -m twin train --no-validate
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import config as C
from . import validate
from .data import load_priors, profile_months, profiles
from .world import redraw
from .world.flow import SCALE, FlowModel, frame_features


def fit_world(seeds=(0, 1, 2, 3, 4)):
    priors, prof = load_priors(), profiles()
    pm = profile_months(priors)
    X, y, g = frame_features(pm), (pm.flow_real / SCALE).values, pm.profile_id.values
    fm = FlowModel.fit(X, y, g, seeds)
    fm.fit_offsets(X, y, g)
    return fm, redraw.IncomeModel.fit(prof), redraw.RentModel.fit(prof)


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 5)
    return x


def main(run_validation=True, log=print):
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fm, income, rent = fit_world()
    fm.save(C.ARTIFACTS / "flow")
    redraw.save(C.ARTIFACTS / "redraw.json", income, rent)
    log(f"world model saved to {C.ARTIFACTS} in {time.time() - t0:.0f}s "
        f"(val NLL per member: {[round(i['val_nll'], 4) for i in fm.info]})")
    if run_validation:
        res = validate.run(fm_final=fm, log=log)
        (C.ARTIFACTS / "validation.json").write_text(json.dumps(to_json(res), indent=1))
        for line in validate.summary_lines(res):
            log(line)
        log(f"acceptance tests {'PASSED' if res['passed'] else 'FAILED'} in {res['seconds']:.0f}s")
        return res
