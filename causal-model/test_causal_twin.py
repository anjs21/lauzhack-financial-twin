"""Accounting, intervention isolation, timing, and repository integration checks."""
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
import unittest

from causal_twin import Assumptions, State, transition, simulate, summary, load_profile


class CausalTwinTests(unittest.TestCase):
    def test_example_and_mechanism_reconciliation(self):
        a = Assumptions()
        work = transition(State(0, D(8000)), 'work', a)
        study = transition(State(0, D(8000)), 'enroll', a)
        self.assertEqual(work.after.cash, D(9200))
        self.assertEqual(study.after.cash, D(-900))
        self.assertEqual(study.after.cash - work.after.cash,
                         study.income - work.income + work.living_costs - study.living_costs
                         + work.tuition - study.tuition + study.scholarship - work.scholarship)

    def test_scholarship_changes_cash_not_progress_or_earnings(self):
        baseline = simulate(8000, Assumptions(), 24, 0)
        grant = simulate(8000, Assumptions(annual_scholarship=5000), 24, 0)
        self.assertEqual(grant[-1].after.cash - baseline[-1].after.cash, D(10000))
        for x, y in zip(baseline, grant):
            self.assertEqual(x.income, y.income)
            self.assertEqual(x.after.study_progress, y.after.study_progress)
            self.assertEqual(x.after.education, y.after.education)

    def test_graduation_delay_and_annual_payment_timing(self):
        trace = simulate(8000, Assumptions(employment_delay_months=3), 28, 0)
        self.assertEqual([x.before.month for x in trace if x.tuition], [0, 12])
        self.assertEqual(trace[23].after.education, 'graduated')
        self.assertEqual(trace[23].income, D(1200))
        self.assertTrue(all(x.after.employment == 'job_seeking' for x in trace[24:27]))
        self.assertEqual(trace[27].after.employment, 'graduate_job')

    def test_intramonth_shortfall_and_exact_top_up(self):
        a = Assumptions(study_income=10000, minimum_reserve=1000)
        trace = simulate(7000, a, 1, 0)
        self.assertGreater(trace[0].after.cash, 0)
        self.assertEqual(summary(trace)['first_shortfall_month'], 1)
        funding = summary(trace)['extra_initial_funding']
        self.assertEqual(funding, D(2000))
        funded = simulate(D(7000) + funding, a, 1, 0)
        self.assertEqual(summary(funded)['minimum_cash'], D(1000))

    def test_delay_enrollment_and_cash_identity(self):
        trace = simulate(8000, Assumptions(), 40, 12)
        self.assertEqual(trace[12].action, 'enroll')
        self.assertEqual([x.before.month for x in trace if x.tuition], [12, 24])
        for step in trace:
            self.assertEqual(step.after.cash, step.before.cash + step.income - step.living_costs
                             - step.tuition + step.scholarship)

    def test_invalid_actions_and_assumptions(self):
        with self.assertRaises(ValueError):
            transition(State(0, D(8000)), 'study', Assumptions())
        with self.assertRaises(ValueError):
            transition(State(0, D(8000), 1, 'studying'), 'work', Assumptions())
        for override in ({'study_months': 0}, {'employment_delay_months': -1},
                         {'graduate_income': float('nan')}, {'work_income': True},
                         {'annual_scholarship': 9000}):
            with self.assertRaises(ValueError):
                replace(Assumptions(), **override)

    def test_repo_inputs_and_monotonic_tuition(self):
        profile, a, ids = load_profile(Path(__file__).resolve().parent.parent / 'Data')
        self.assertEqual(profile['display_name'], 'Alexia')
        self.assertEqual(a.study_months, 24)
        self.assertEqual(a.annual_tuition, 8000)
        self.assertEqual(len(ids), 2)
        low = summary(simulate(8000, a, 120, 0))
        high = summary(simulate(8000, replace(a, annual_tuition=10000), 120, 0))
        self.assertEqual(low['closing_cash'] - high['closing_cash'], D(4000))
        self.assertGreaterEqual(high['extra_initial_funding'], low['extra_initial_funding'])


if __name__ == '__main__':
    unittest.main()
