# State-pair association analysis

Profiles: 1200. Tests: 181. BH-adjusted q < 0.05: 81.

One observation per profile, not 28,800 independent monthly observations. All numeric pairs have Pearson (linear), Spearman (rank), and Kendall (rank concordance) tests. Numeric/category pairs use Kruskal-Wallis with epsilon-squared effect. Category pairs use chi-square and bias-corrected Cramer's V; sparse tables use 1,999 seeded permutations. Two-sided tests where applicable. BH adjustment is applied jointly to all reported p-values; exploratory control is not a guarantee under arbitrary test dependence.

Synthetic records are not a random sample of Swiss households. P-values describe tests under statistical assumptions and should not support population claims or causal conclusions. Small q-values do not imply large effects; non-significance does not prove independence.

Opening amounts are January 2025 snapshots. Mean/SD net flows, final cash, and shortfall are retrospective 2025–2026 summaries, not information available at the initial decision. Do not use them as initial-state predictors in a historical policy evaluation.

Final cash = opening cash + 24 × mean monthly net. Debt presence is derived from debt amount, and shortfall from the cash trajectory. These mechanically linked pairs are flagged in the results. Repeated country/currency/as-of-date fields and identifiers are excluded; categorical labels are not assigned arbitrary numeric codes for correlations.

Stratified employment results help reveal pooled-versus-within-group differences but do not identify confounding or causal effects. No action outcomes or causal transition probabilities are estimated.

## Strongest pooled numeric rank associations

| Pair | Spearman rho | BH q | Caveat |
|---|---:|---:|---|
| reference_monthly_net_income_chf / mean_monthly_net_chf | 0.967 | 0 |  |
| mean_monthly_net_chf / final_cash_chf | 0.853 | 0 | Final cash = opening cash + 24 * mean monthly net; mechanically coupled. |
| reference_monthly_net_income_chf / final_cash_chf | 0.827 | 3.2e-300 |  |
| age_at_start / opening_investments_chf | 0.690 | 1.59e-169 |  |
| reference_monthly_net_income_chf / sd_monthly_net_chf | 0.651 | 1.82e-144 |  |
| mean_monthly_net_chf / sd_monthly_net_chf | 0.651 | 1.95e-144 |  |
| sd_monthly_net_chf / final_cash_chf | 0.544 | 2.25e-92 |  |
| opening_cash_chf / final_cash_chf | 0.470 | 2.34e-66 | Final cash = opening cash + 24 * mean monthly net; mechanically coupled. |
| age_at_start / opening_cash_chf | 0.437 | 2.07e-56 |  |
| opening_cash_chf / opening_investments_chf | 0.336 | 2.33e-32 |  |
| age_at_start / reference_monthly_net_income_chf | -0.255 | 9.61e-19 |  |
| age_at_start / mean_monthly_net_chf | -0.244 | 3.7e-17 |  |

## Reproduce

From the repository root:

```bash
python analysis/state-correlations/analyze_state_pairs.py
```

Dependencies are listed in ../requirements.txt. All outputs are in this results directory.

Methods: https://docs.scipy.org/doc/scipy/reference/stats.html

## Selected findings and MDP implications

- Reference income vs mean monthly net flow: Pearson 0.972, Spearman 0.967, Kendall 0.841. Very strong association across these synthetic profiles; these measures do not identify the effect of accepting a job.
- Age vs opening investments: Spearman 0.690. Age and starting wealth are associated, but increasing simulated age must not automatically create investment holdings.
- Household type vs reference rent: Kruskal-Wallis epsilon-squared 0.560. Household-specific housing assumptions are worth considering; changing household type does not guarantee this observed group difference.
- Employment vs cash shortfall: bias-corrected Cramer V 0.765; permutation p=0.0005, BH q=0.00118. Shortfall is a retrospective trajectory label, not an initial state fact.
- Opening cash vs opening debt: Spearman 0.00094, BH q=0.985. No detected monotonic association; this does not prove independence.
- Pooled age-income Spearman is -0.255 (BH q=9.61e-19). Within each employment group, rho ranges from -0.097 to 0.113 and all BH q-values exceed 0.45. Employment composition may explain the pooled pattern; stratification alone does not establish a causal explanation.

Numerical p/q values stored as zero reflect floating-point underflow. Do not treat the number of significant tests as a number of independent discoveries. Cross-sectional associations cannot replace action-conditioned causal transitions.
