# FAQ — Troubleshooting and Common Mistakes

**LauzHack 2026 | Practical guidance**

### 1. The CSV files do not join correctly.
Check exact `profile_id` values, column names, data types, and missing keys. Avoid assuming every file uses the same column order.

### 2. My cash balance does not reconcile.
Inspect signed transaction amounts, duplicated transactions, transfers, fees, and opening-balance dates.

### 3. Net worth drops when moving cash to investments.
Treat the transfer as a movement between asset accounts, not consumption.

### 4. My annual and monthly numbers disagree.
Convert all flows and interest rates to a consistent time step before calculating.

### 5. The LLM produces inconsistent numbers.
Have a calculation engine compute results. Ask the LLM to explain verified outputs, not invent them.

### 6. Two scenarios look identical.
Confirm that the intervention actually changes the state-transition inputs and that cached outputs are refreshed.

### 7. The game shows different values from the dashboard.
Connect both views to the same financial state and test the event update logic.

### 8. My simulation produces negative savings.
Show the liquidity shortfall and any financing assumption. Do not silently clamp negative cash to zero.

### 9. My Monte Carlo results change every run.
Use a documented random seed for reproducible tests; still explain sampling uncertainty.

### 10. My RL agent finds unrealistic strategies.
Inspect reward hacking, action constraints, simulator loopholes, and environment realism.

### 11. The optimizer returns an impossible solution.
Check constraint definitions, units, bounds, solver status, and whether the model is feasible.

### 12. My public dataset has missing years.
Document gaps and avoid inventing historical observations without a clearly labelled imputation method.

### 13. An API link fails.
Check whether it is a direct download, an API response, or an interactive export page; verify rate limits and required parameters.

### 14. GitHub does not render my equations.
In Markdown files, use `$...$` for inline math and `$$...$$` on separate lines for display math.

### 15. GitHub does not show an empty folder.
Create a file such as `folder/README.md`; Git tracks files, not empty directories.

### 16. A download does not work on mobile.
Try the source page's download/export control or GitHub's desktop view. Large datasets are easier to upload from a computer.

### 17. Can we claim results are forecasts?
Avoid such claims unless a separately validated forecasting model is actually implemented. A Financial Twin can explore scenarios without forecasting.

### 18. What should we do if assumptions are uncertain?
Show the assumptions and optionally offer sensitivity, stress, or probabilistic scenario analysis.

### 19. How do we make the demo credible?
Show inputs, interventions, calculations, comparisons, and at least one validation check.

### 20. Where can we find more help?
Consult `docs/`, `examples/`, `resources/`, and `evaluation/`. For binding event requirements, contact the organizers.
