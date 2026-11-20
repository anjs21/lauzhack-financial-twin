Optional Financial Twin Starter Kit
LauzHack 2026 — AI Financial Twin Challenge
This is a minimal, educational, standard-library-only Python example. It is not a required architecture, financial advice, or a forecasting system. Teams are free to replace it entirely.

What it does
- Loads a fictional profile and three hypothetical scenarios.
- Simulates monthly cash balances over ten years.
- Compares the final cash position and lowest monthly cash balance.
- Flags months when cash is negative.
- Uses explicit editable assumptions; no external APIs or paid services.
Important: This deliberately models cash only, not complete net worth. It does not model investment returns, debt, taxes, pension rights, inflation, exchange rates, or stochastic uncertainty. Salary and spending figures are invented teaching assumptions, not official Swiss statistics.
Quick start
Requires Python 3.10+.

From the repository root:
python starter-kit/run_example.py
python -m unittest discover -s starter-kit -p "test_*.py"
Or enter the directory and run:
cd starter-kit
python run_example.py
python -m unittest discover -p "test_*.py"
No pip install is required.
Files
File	Role
QUICKSTART.md	Walkthrough and customization
financial_twin.py	Deterministic monthly simulation engine
run_example.py	Command-line scenario comparison
sample_profile.json	Fictional Alexia profile
scenario_config.json	Editable assumptions for three scenarios
test_financial_twin.py	Unit tests and accounting checks
requirements.txt	Empty: Python standard library only
.gitignore	Local cache and secret exclusions


Modelling convention
A month is indexed from 0 to 119. At the beginning of each month, any one-off event occurs; then the model adds monthly income and subtracts monthly expenses:
\[
C_{t+1}=C_t + I_t-E_t-K_t
\]
where $K_t$ is any one-off cost in month $t$.

A scenario consists of phases with [start_month, end_month) intervals that must cover the entire 120-month horizon without gaps or overlaps.

The three sample scenarios are fictional illustrations. They are not recommendations or predictions.
Suggested extensions
Add income tax, assets, liabilities, education financing, inflation, multi-currency accounts, probability distributions, causal interventions, optimization, agents, games, or a 3D interface. Keep the financial accounting auditable.
Data and rights
This starter kit uses only fictional example values. External datasets and software are optional; consult resources/ for links and license guidance. See the repository's NOTICE.md and licensing documentation for the rights applicable to original challenge materials. No third-party data is bundled here.

Don't predict the future. Make it explorable.
