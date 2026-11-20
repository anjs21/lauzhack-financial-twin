"""Basic accounting and validation tests for the illustrative simulator."""
import unittest

from financial_twin import simulate


class FinancialTwinTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"opening_cash_chf": 8000, "horizon_months": 2}
        self.scenario = {
            "name": "Test",
            "phases": [{"start_month": 0, "end_month": 2,
                        "monthly_income_chf": 1200, "monthly_expenses_chf": 1800}],
            "one_off_costs": [{"month": 0, "amount_chf": 500}],
        }

    def test_cash_reconciliation(self):
        result = simulate(self.profile, self.scenario)
        self.assertEqual(result.cash_by_month_chf, (6900, 6300))
        self.assertEqual(result.closing_cash_chf, 6300)
        self.assertEqual(result.minimum_cash_chf, 6300)

    def test_negative_cash_flag(self):
        self.profile["opening_cash_chf"] = 0
        result = simulate(self.profile, self.scenario)
        self.assertEqual(result.first_negative_month, 0)

    def test_gap_rejected(self):
        self.scenario["phases"][0]["end_month"] = 1
        with self.assertRaises(ValueError):
            simulate(self.profile, self.scenario)

    def test_overlap_rejected(self):
        self.scenario["phases"].append({
            "start_month": 1, "end_month": 2,
            "monthly_income_chf": 0, "monthly_expenses_chf": 0})
        with self.assertRaises(ValueError):
            simulate(self.profile, self.scenario)

    def test_negative_one_off_cost_rejected(self):
        self.scenario["one_off_costs"][0]["amount_chf"] = -1
        with self.assertRaises(ValueError):
            simulate(self.profile, self.scenario)


if __name__ == "__main__":
    unittest.main()
