"""Descriptive pairwise state associations; not causal or transition estimation.
Run: python analysis/state-correlations/analyze_state_pairs.py
Dependencies: numpy, scipy, pandas, matplotlib.
"""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'results'
NUMERIC = ['age_at_start', 'opening_cash_chf', 'opening_investments_chf',
           'opening_debt_chf', 'reference_monthly_net_income_chf',
           'reference_monthly_rent_chf', 'mean_monthly_net_chf',
           'sd_monthly_net_chf', 'final_cash_chf']
CATEGORICAL = ['employment_status', 'household_type', 'city', 'has_debt', 'cash_shortfall']


def load():
    def read(name):
        return pd.read_csv(ROOT / 'Data' / f'{name}.csv')
    df = read('synthetic-profiles').set_index('profile_id')
    assert df.index.is_unique
    assets = read('synthetic-assets')
    loans = read('synthetic-liabilities')
    balances = read('synthetic-cash-balances').sort_values(['profile_id', 'month'])
    assert not balances.duplicated(['profile_id', 'month']).any()
    assert (balances.groupby('profile_id').size() == 24).all()
    assert np.allclose(balances.opening_cash_chf + balances.net_transactions_chf,
                       balances.closing_cash_chf, atol=.001)
    df['opening_investments_chf'] = assets[assets.asset_type == 'investment_portfolio'].groupby('profile_id').value_chf.sum().reindex(df.index, fill_value=0)
    df['opening_debt_chf'] = loans.groupby('profile_id').opening_principal_chf.sum().reindex(df.index, fill_value=0)
    g = balances.groupby('profile_id')
    df['mean_monthly_net_chf'] = g.net_transactions_chf.mean()
    df['sd_monthly_net_chf'] = g.net_transactions_chf.std()
    df['final_cash_chf'] = g.closing_cash_chf.last()
    df['cash_shortfall'] = g.closing_cash_chf.min().lt(0).map({True: 'yes', False: 'no'})
    df['has_debt'] = df.opening_debt_chf.gt(0).map({True: 'yes', False: 'no'})
    assert np.allclose(df.final_cash_chf, df.opening_cash_chf + 24 * df.mean_monthly_net_chf)
    assert not df[NUMERIC + CATEGORICAL].isna().any().any()
    return df


def bh(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    results = []
    def add(x, y, method, effect_name, effect, p, n, note='', scope='all profiles'):
        results.append(dict(state_x=x, state_y=y, scope=scope, method=method,
                            effect_name=effect_name, effect=float(effect), p_value=float(p),
                            n=int(n), caveat=note))
    for x, y in combinations(NUMERIC, 2):
        for method, function in [('Pearson', stats.pearsonr), ('Spearman', stats.spearmanr), ('Kendall', stats.kendalltau)]:
            r = function(df[x], df[y])
            note = 'Final cash = opening cash + 24 * mean monthly net; mechanically coupled.' if 'final_cash_chf' in (x, y) and {'opening_cash_chf', 'mean_monthly_net_chf'} & {x, y} else ''
            add(x, y, method, method, r.statistic, r.pvalue, len(df), note)
    for x in CATEGORICAL:
        for y in NUMERIC:
            groups = [v[y].to_numpy() for _, v in df.groupby(x)]
            h, p = stats.kruskal(*groups)
            # Epsilon-squared rank effect, clamped at zero for finite-sample noise.
            effect = max(0, (h - len(groups) + 1) / (len(df) - len(groups)))
            note = 'Debt indicator is defined from this amount.' if x == 'has_debt' and y == 'opening_debt_chf' else ''
            if x == 'cash_shortfall':
                note = 'Shortfall is derived from the same cash trajectory; retrospective outcome.'
            add(x, y, 'Kruskal-Wallis', 'epsilon_squared', effect, p, len(df), note)
    rng = np.random.default_rng(20261121)
    for x, y in combinations(CATEGORICAL, 2):
        a, b = pd.factorize(df[x])[0], pd.factorize(df[y])[0]
        rows, cols = a.max()+1, b.max()+1
        observed = np.bincount(a*cols+b, minlength=rows*cols).reshape(rows, cols)
        chi, p, _, expected = stats.chi2_contingency(observed, correction=False)
        method = 'Chi-square'
        note = ''
        if (expected < 5).any():
            # Fixed margins, permutation calibration for sparse tables.
            count = 0
            for _ in range(1999):
                shuffled = np.bincount(a*cols+rng.permutation(b), minlength=rows*cols).reshape(rows, cols)
                perm_stat = ((shuffled-expected)**2/expected).sum()
                count += perm_stat >= chi - 1e-12
            p = (count+1)/2000
            method = 'Chi-square permutation (1999)'
            note = 'Sparse expected counts; seeded permutation p-value (resolution 0.0005).'
        # Bias-corrected Cramer's V.
        n = len(df)
        phi = max(0, chi/n - (rows-1)*(cols-1)/(n-1))
        denominator = min(rows-1-(rows-1)**2/(n-1), cols-1-(cols-1)**2/(n-1))
        v = np.sqrt(phi/denominator) if denominator > 0 else 0
        add(x, y, method, 'bias_corrected_cramers_v', v, p, n, note)
    # Compare pooled associations with within-employment associations, not causal adjustment.
    pairs = [('age_at_start', 'reference_monthly_net_income_chf'),
             ('reference_monthly_net_income_chf', 'mean_monthly_net_chf'),
             ('opening_cash_chf', 'mean_monthly_net_chf')]
    for status, subset in df.groupby('employment_status'):
        for x, y in pairs:
            if subset[x].nunique() < 2 or subset[y].nunique() < 2:
                continue
            r = stats.spearmanr(subset[x], subset[y])
            add(x, y, 'Spearman', 'Spearman', r.statistic, r.pvalue, len(subset),
                'Descriptive stratification only; not a causal effect estimate.', f'employment={status}')
    table = pd.DataFrame(results)
    assert np.isfinite(table[['effect', 'p_value']].to_numpy()).all()
    table['q_value_bh'] = bh(table.p_value)
    table['significant_fdr_005'] = table.q_value_bh < .05
    table.to_csv(OUT / 'pairwise_tests.csv', index=False)
    df[NUMERIC+CATEGORICAL].to_csv(OUT / 'profile_states.csv')
    pd.concat([df.groupby('employment_status')[NUMERIC].median(),
               df.groupby('employment_status').size().rename('n')], axis=1).to_csv(OUT / 'employment_medians.csv')
    matrix = df[NUMERIC].corr(method='spearman')
    matrix.to_csv(OUT / 'spearman_matrix.csv')
    labels = [s.replace('_chf', '').replace('reference_monthly_', '').replace('_', ' ') for s in NUMERIC]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=65, ha='right')
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{matrix.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title('Synthetic profile states: Spearman correlation (n=1,200)\nHistory features use all 24 months; association is not causation')
    fig.colorbar(im, ax=ax, label='Spearman rho')
    fig.tight_layout()
    fig.savefig(OUT / 'spearman_heatmap.png', dpi=160)
    plt.close(fig)
    metadata = dict(profiles=len(df), tests=len(table), significant=int(table.significant_fdr_005.sum()),
                    scipy=scipy.__version__, pandas=pd.__version__, numpy=np.__version__, seed=20261121)
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = ['# State-pair association analysis', '', f"Profiles: {len(df)}. Tests: {len(table)}. BH-adjusted q < 0.05: {metadata['significant']}.", '',
      'One observation per profile, not 28,800 independent monthly observations. All numeric pairs have Pearson (linear), Spearman (rank), and Kendall (rank concordance) tests. Numeric/category pairs use Kruskal-Wallis with epsilon-squared effect. Category pairs use chi-square and bias-corrected Cramer\'s V; sparse tables use 1,999 seeded permutations. Two-sided tests where applicable. BH adjustment is applied jointly to all reported p-values; exploratory control is not a guarantee under arbitrary test dependence.', '',
      'P-values or q-values printed as 0 are floating-point underflow, not exact zero probabilities. Cramer V, epsilon-squared, and signed correlations have different interpretations and should not be compared as identical scales.', '',
      'Synthetic records are not a random sample of Swiss households. P-values describe tests under statistical assumptions and should not support population claims or causal conclusions. Small q-values do not imply large effects; non-significance does not prove independence.', '',
      'Opening amounts are January 2025 snapshots. Mean/SD net flows, final cash, and shortfall are retrospective 2025–2026 summaries, not information available at the initial decision. Do not use them as initial-state predictors in a historical policy evaluation.', '',
      'Final cash = opening cash + 24 × mean monthly net. Debt presence is derived from debt amount, and shortfall from the cash trajectory. These mechanically linked pairs are flagged in the results. Repeated country/currency/as-of-date fields and identifiers are excluded; categorical labels are not assigned arbitrary numeric codes for correlations.', '',
      'Stratified employment results help reveal pooled-versus-within-group differences but do not identify confounding or causal effects. No action outcomes or causal transition probabilities are estimated.', '',
      '## Strongest pooled numeric rank associations', '', '| Pair | Spearman rho | BH q | Caveat |', '|---|---:|---:|---|']
    subset = table[(table.method == 'Spearman') & (table.scope == 'all profiles')].copy()
    subset = subset.loc[subset.effect.abs().sort_values(ascending=False).index].head(12)
    for r in subset.itertuples():
        lines.append(f'| {r.state_x} / {r.state_y} | {r.effect:.3f} | {r.q_value_bh:.3g} | {r.caveat} |')
    lines += ['', '## Reproduce', '', 'From the repository root:', '', '```bash',
              'python analysis/state-correlations/analyze_state_pairs.py', '```', '',
              'Dependencies are listed in ../requirements.txt. All outputs are in this results directory.', '',
              'Methods: https://docs.scipy.org/doc/scipy/reference/stats.html', '']
    lines += [
        '## Selected findings and MDP implications\n\n- Reference income vs mean monthly net flow: Pearson 0.972, Spearman 0.967, Kendall 0.841. Very strong association across these synthetic profiles; these measures do not identify the effect of accepting a job.\n- Age vs opening investments: Spearman 0.690. Age and starting wealth are associated, but increasing simulated age must not automatically create investment holdings.\n- Household type vs reference rent: Kruskal-Wallis epsilon-squared 0.560. Household-specific housing assumptions are worth considering; changing household type does not guarantee this observed group difference.\n- Employment vs cash shortfall: bias-corrected Cramer V 0.765; permutation p=0.0005, BH q=0.00118. Shortfall is a retrospective trajectory label, not an initial state fact.\n- Opening cash vs opening debt: Spearman 0.00094, BH q=0.985. No detected monotonic association; this does not prove independence.\n- Pooled age-income Spearman is -0.255 (BH q=9.61e-19). Within each employment group, rho ranges from -0.097 to 0.113 and all BH q-values exceed 0.45. Employment composition may explain the pooled pattern; stratification alone does not establish a causal explanation.\n\nNumerical p/q values stored as zero reflect floating-point underflow. Do not treat the number of significant tests as a number of independent discoveries. Cross-sectional associations cannot replace action-conditioned causal transitions.\n'
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(metadata))
    print(subset[['state_x','state_y','effect','q_value_bh']].to_string(index=False))


if __name__ == '__main__':
    main()
