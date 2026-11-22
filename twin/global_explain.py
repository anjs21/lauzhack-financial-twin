"""Global grouped permutation importance for the saved population flow network.

Computes NLL change by observed year using person-level permutations. Final-model
training data are used, so this is descriptive, not held-out generalisation.
"""
from __future__ import annotations

import numpy as np
import torch
import pandas as pd
from scipy.special import logsumexp
from .data import profile_months
from .world.flow import SCALE, frame_features

GROUPS = {
    'income': ('income_real',),
    'rent': ('rent_real',),
    'employment': ('mode', 'self_employed', 'pensioner'),
    'city': ('city_idx',),
    'season': ('moy',),
}


def ensemble_nll(model, frame):
    X = frame_features(frame)
    y = frame.flow_real.to_numpy(dtype=float) / SCALE
    member_log_likelihoods = []
    for member in range(len(model.members)):
        weights, means, sigmas = model.params(X, member)
        terms = np.log(np.maximum(weights, np.finfo(float).tiny)) - np.log(sigmas)
        terms -= .5*((y[:, None]-means)/sigmas)**2 + .5*np.log(2*np.pi)
        member_log_likelihoods.append(logsumexp(terms, axis=1))
    return -logsumexp(np.stack(member_log_likelihoods), axis=0) + np.log(len(model.members))


def grouped_permutation(frame, group, rng):
    """One consistent donor for each person's profile variables, within a year.

    Seasonal permutation rotates each person's 12 month labels, preserving the
    number of each month, and frame_features recomputes all interactions.
    """
    altered = frame.copy()
    codes, people = pd.factorize(frame.profile_id, sort=True)
    if group == 'season':
        shifts = rng.integers(1, 12, size=len(people))
        altered['moy'] = (frame.moy.to_numpy() - 1 + shifts[codes]) % 12 + 1
    else:
        donor = rng.permutation(len(people))
        lookup = pd.MultiIndex.from_arrays([people[donor[codes]], frame.t.to_numpy()])
        source = frame.set_index(['profile_id', 't'])
        for col in GROUPS[group]:
            altered[col] = source[col].reindex(lookup).to_numpy()
            if pd.isna(altered[col]).any():
                raise ValueError('Every profile must have matching months for permutation')
    return altered


def global_importance(model, repeats=3, seed=20261121):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    frame = profile_months().copy()
    frame['year'] = C_START_YEAR + frame.t // 12
    rng = np.random.default_rng(seed)
    results = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for year, year_data in frame.groupby('year', sort=True):
            base = float(ensemble_nll(model, year_data).mean())
            for group in GROUPS:
                changes = []
                for _ in range(repeats):
                    shuffled = grouped_permutation(year_data, group, rng)
                    changes.append(float(ensemble_nll(model, shuffled).mean()-base))
                results.append(dict(year=int(year), feature_group=group,
                    delta_nll_mean=float(np.mean(changes)), delta_nll_repeats=changes,
                    baseline_nll=base, rows=len(year_data),
                    profiles=int(year_data.profile_id.nunique())))
    finally:
        torch.set_num_threads(previous_threads)
    return dict(method='Grouped permutation importance: change in mean negative log likelihood of observed net flows',
        seed=seed, repeats=repeats, years=results,
        limitations=[
            'The saved final model was trained on these synthetic profiles and years. This is an in-sample diagnostic, not held-out feature importance or a population claim.',
            'Income, rent, employment and city are shuffled across people and using the same donor for each person throughout each observed year; season uses a random cyclic shift per person.',
            'Permutations can create implausible combinations and break correlations. Importance is model reliance under this perturbation, not a causal effect.',
            'The importance score can be negative, and repeated permutations show only shuffle variability, not confidence intervals.',
            'These scores cover observed 2025 and 2026 flows only, exclude personal offsets and other simulator mechanisms, and do not establish importance in future simulated years.',
        ])


from .config import START_YEAR as C_START_YEAR
