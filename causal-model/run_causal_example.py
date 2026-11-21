"""Compare causal financial scenarios using repository inputs and explicit assumptions."""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

from causal_twin import load_profile, simulate, summary, sensitivity

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='P00001')
    parser.add_argument('--months', type=int, default=120)
    parser.add_argument('--assumptions', type=Path, default=HERE / 'causal_assumptions.json')
    parser.add_argument('--export', type=Path, help='Write monthly causal mechanisms to a CSV')
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()
    profile, assumptions, source_ids = load_profile(HERE.parent / 'Data', args.profile)
    config = json.loads(args.assumptions.read_text(encoding='utf-8'))
    assumptions = replace(assumptions, **config['parameters'])
    print(f"Causal Financial Twin — {profile['display_name']} ({args.profile})")
    print(f"Opening cash: CHF {profile['opening_cash_chf']}; month 1 starts {profile['as_of_date']}")
    print(f"CSV scenario inputs: {', '.join(source_ids)}")
    print('Effective assumptions:', json.dumps(asdict(assumptions), sort_keys=True))
    print('Other inputs are illustrative assumptions, not estimated causal effects.')
    print('Negative cash is an unfunded projection. No automatic loans or investment sales.\n')
    scenarios = [('Work now', None), ('Study now', 0)]
    traces = {}
    for name, enrollment in scenarios:
        trace = simulate(profile['opening_cash_chf'], assumptions, args.months, enrollment)
        traces[name] = trace
        result = summary(trace)
        print(f"{name}: closing CHF {result['closing_cash']:,.2f}; minimum CHF {result['minimum_cash']:,.2f}; "
              f"extra initial funding for reserve CHF {result['extra_initial_funding']:,.2f}; "
              f"first shortfall month: {result['first_shortfall_month'] or 'none'}")
    work, study = traces['Work now'], traces['Study now']
    differences = {
        'earnings': sum(s.income - w.income for w, s in zip(work, study)),
        'living costs': sum(w.living_costs - s.living_costs for w, s in zip(work, study)),
        'tuition': sum(w.tuition - s.tuition for w, s in zip(work, study)),
        'scholarship': sum(s.scholarship - w.scholarship for w, s in zip(work, study)),
    }
    print('\nStudy minus work closing-cash difference by accounting mechanism:')
    for mechanism, value in differences.items():
        print(f'  {mechanism}: CHF {value:,.2f}')
    print(f'  Total: CHF {sum(differences.values()):,.2f}')
    print('This decomposition follows the specified model; it is not an empirical causal estimate.')
    if args.sensitivity:
        print('\nOne-at-a-time sensitivity (fixed policies):')
        for parameter, value, result, delta in sensitivity(profile['opening_cash_chf'], assumptions, args.months):
            print(f"  {parameter}={value}: study-work CHF {delta:,.2f}; "
                  f"required initial top-up CHF {result['extra_initial_funding']:,.2f}")
    if args.export:
        fields = ['scenario', 'month', 'action', 'opening_cash', 'income', 'living_costs', 'tuition',
                  'scholarship', 'cash_after_upfront_costs', 'closing_cash', 'minimum_cash',
                  'reserve_gap', 'education', 'study_progress', 'employment']
        with args.export.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name, trace in traces.items():
                for step in trace:
                    writer.writerow(dict(scenario=name, month=step.before.month + 1, action=step.action,
                        opening_cash=step.before.cash, income=step.income, living_costs=step.living_costs,
                        tuition=step.tuition, scholarship=step.scholarship,
                        cash_after_upfront_costs=step.cash_after_upfront_costs, closing_cash=step.after.cash,
                        minimum_cash=step.minimum_cash, reserve_gap=step.reserve_gap,
                        education=step.after.education, study_progress=step.after.study_progress,
                        employment=step.after.employment))
        print(f'\nMonthly traces written to {args.export}')


if __name__ == '__main__':
    main()
