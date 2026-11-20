"""Run the optional LauzHack Financial Twin example from any working directory."""
import json
from pathlib import Path

from financial_twin import simulate

HERE = Path(__file__).resolve().parent


def main() -> None:
    profile = json.loads((HERE / "sample_profile.json").read_text(encoding="utf-8"))
    config = json.loads((HERE / "scenario_config.json").read_text(encoding="utf-8"))
    print(f"Financial Twin — {profile['name']} (fictional)")
    print(f"Opening cash: CHF {profile['opening_cash_chf']:,.2f}")
    print("Illustrative assumptions only; cash balances are not forecasts.\n")
    results = [simulate(profile, s) for s in config["scenarios"]]
    print(f"{'Scenario':<24} {'Closing cash':>16} {'Min cash':>16} {'First shortfall':>17}")
    print("-" * 78)
    for r in results:
        shortfall = "None" if r.first_negative_month is None else f"Month {r.first_negative_month + 1}"
        print(f"{r.scenario_name:<24} CHF {r.closing_cash_chf:>12,.2f} "
              f"CHF {r.minimum_cash_chf:>12,.2f} {shortfall:>17}")
    print("\nA negative balance indicates an unfunded cash shortfall, not an assumed loan.")


if __name__ == "__main__":
    main()
