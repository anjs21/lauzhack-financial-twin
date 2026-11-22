# Sensitivity explorer

From the repository root, start the local dashboard:

```bash
python -B analysis/sensitivity/dashboard.py
```

Open http://127.0.0.1:8765. Select a profile and use **Run analysis for selected profile** if it has not been analysed yet. A full run uses 1,000 paired paths for the baseline and individual stresses, and 300 paired paths for each grid cell. The dashboard runs one analysis at a time. Stop the server with Ctrl+C.

Alternatively run an analysis from the command line:

```bash
python -B analysis/sensitivity/run_sensitivity.py --profile P00004 --paths 1000 --grid-paths 300 --seed 20261122
```

Results are saved under `analysis/sensitivity/results/profiles/<profile-id>/` as JSON, CSV and a Markdown report. Running the same profile again replaces that profile's generated files. The original Alexia result remains available as a fallback when no new P00001 run exists.

The baseline assumes CHF 8,000 annual tuition for all profiles as a common stress-test assumption. Study and work are opening intentions, subject to action masks and subsequent admission or job offers; the action selection rates reveal when an intention is unavailable or redundant. Goals use the simulator's fixed December 2029 deadline. The heatmap shows combined scenarios, not isolated interaction effects. Intervals quantify Monte Carlo error conditional on the model, not real-world forecast uncertainty or causal effects.
