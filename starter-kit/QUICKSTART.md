# Quickstart — Run and Extend the Starter Kit

## 1. Install Python

Use Python 3.10 or newer. This example uses only built-in modules.

## 2. Download the files

Keep all starter-kit files together in the `starter-kit/` folder.

## 3. Run the example

From the repository root:

```bash
python starter-kit/run_example.py
```

The command prints a comparison of three fictional scenarios and their liquidity constraints.

## 4. Run tests

```bash
python -m unittest discover -s starter-kit -p "test_*.py"
```

## 5. Modify assumptions

Edit `scenario_config.json`:

- `monthly_income_chf` — monthly net income **assumption**
- `monthly_expenses_chf` — monthly total spending **assumption**
- `one_off_costs` — costs applied at the beginning of specified months
- `phases` — consecutive time intervals covering the full horizon

Edit `sample_profile.json` to change the opening cash balance or time horizon.

## 6. Explore a counterfactual

Change one value (e.g. education costs), run the example again, and compare results with the original scenario. Keep a record of the assumptions.

## 7. Understand limitations

The engine models **cash only**. It does not estimate the real-world likelihood of a scenario or provide personal advice. Negative cash is shown as a **funding shortfall**; the model does not invent a loan to cover it.

## 8. Build your own Financial Twin

You may extend this code, replace the interface, use another language, build a game, introduce AI agents, or create a different modelling architecture. No particular technology is mandatory.

## GitHub mobile tip

Create a file by entering its full path, such as `starter-kit/run_example.py`, in GitHub's **Add file → Create new file** interface. Paste its content and commit.
