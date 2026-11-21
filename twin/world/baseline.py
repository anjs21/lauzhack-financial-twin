"""The hand-calibrated simulator from analysis/action_importance, as a monthly-flow baseline.

It reuses that script's calibration (calibrate, make_state) and its monthly net-flow
formula unchanged, so the flow network is judged against exactly the model it replaces.
No debt service and no overdraft interest, as in the ledger.
"""
from __future__ import annotations

import sys

import numpy as np

from .. import config as C

sys.path.insert(0, str(C.ROOT / "analysis" / "action_importance"))
import action_importance as ai  # noqa: E402


def baseline_flows(years=(2025, 2026), t_offset=0, T=24, K=200, seed=C.SEED):
    """Calibrate on the ledger rows of `years`; return (profile ids, flows P x K x T in CHF)."""
    prof, led = ai.load()
    cal, other, base_se = ai.calibrate(prof, led[led.yr.isin(years)])
    st = ai.make_state(prof, other, base_se)
    P = st.n
    g, pi = cal["drift"]["pay_growth_per_year"], cal["drift"]["cost_inflation_per_year"]
    offset = st.base_se[:, None] * np.random.default_rng([seed, 999_999]).standard_normal((P, K))
    lo, hi = cal["shock"]["size_uniform_chf"]
    blo, bhi = cal["december_bonus"]["size_share_of_income_uniform"]
    code = st.code
    flows = np.empty((P, K, T))
    for t in range(T):
        mon = (t_offset + t) % 12 + 1
        rng = np.random.default_rng([seed, t])
        z, u1, u2, u3, u4 = (rng.standard_normal((P, K)), rng.random((P, K)), rng.random((P, K)),
                             rng.random((P, K)), rng.random((P, K)))
        gy, gc = (1 + g) ** ((t_offset + t) / 12), (1 + pi) ** ((t_offset + t) / 12)
        Y = st.Y[:, None] * gy
        p_sh = cal["shock"]["p_feb_aug"] if mon in (2, 8) else cal["shock"]["p_regular_month"]
        net = (Y - gc * (st.R[:, None] + st.O[:, None]) - gc * (u1 < p_sh) * (lo + (hi - lo) * u2)
               + (cal["_sig_abs"][code][:, None] + cal["_sig_rel"][code][:, None] * Y) * z + offset)
        if mon == 12:
            net += (u3 < cal["_p_bonus"][code][:, None]) * Y * (blo + (bhi - blo) * u4)
        if mon == 2:
            net -= gc * cal["_fee_feb"][code][:, None]
        if mon == 9:
            net -= gc * cal["_fee_sep"][code][:, None]
        flows[:, :, t] = net
    return st.pid, flows
