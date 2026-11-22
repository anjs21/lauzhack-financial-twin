import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

FILE=Path(__file__).with_name('run_explanations.py')
spec=importlib.util.spec_from_file_location('policy_explanations',FILE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PolicyExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.people,cls.persona,cls.plan,cls.entry,cls.entries,cls.prov=m.read_input('example',None,2)
        cls.world=m.World.load()

    def test_opening_counterfactual_and_plan_trigger(self):
        res=m.policy_at_opening(self.world,self.people,self.persona,self.plan,[3000,12000],99)
        lo,hi=res['cash_comparisons']
        self.assertLess(lo['with_plan']['cash_months'],1)
        self.assertGreater(hi['with_plan']['cash_months'],1)
        self.assertGreater(lo['with_plan']['probabilities']['housing'][0],hi['with_plan']['probabilities']['housing'][0])
        for row in (lo,hi):
            for head in m.HEADS:
                self.assertAlmostEqual(sum(row['with_plan']['probabilities'][head]),1)
                for prob,valid in zip(row['with_plan']['probabilities'][head],row['with_plan']['available'][head]):
                    if not valid: self.assertEqual(prob,0)
        self.assertAlmostEqual(res['change_low_to_high']['housing']['change_pp'][0],
                               100*(hi['with_plan']['probabilities']['housing'][0]-lo['with_plan']['probabilities']['housing'][0]))

    def test_live_samples_use_planner_and_report_frequency_uncertainty(self):
        plan=self.plan
        class Stub:
            max_rounds=1
            def __init__(self,*args,**kwargs): pass
            def draft(self,people,persona,wishes):
                return SimpleNamespace(plan=plan,signed=True)
        with patch.object(m.llm_provider,'ready',return_value=True), patch.object(m.llm_provider,'model',return_value='test-model'), patch.object(m,'PlannerAgent',Stub):
            out=m.sampled_plan_comparison(self.world,self.people,self.persona,'same wishes',3000,12000,2,2,9)
        self.assertEqual(out['model'],'test-model')
        self.assertEqual(out['presence_effects']['housing:bigger']['change_pp'],0)
        self.assertLess(out['presence_effects']['housing:bigger']['lower_pp'],0)
        self.assertGreater(out['presence_effects']['housing:bigger']['upper_pp'],0)
        for g in out['groups']:
            x=g['step_presence_frequency']['housing:bigger']
            self.assertEqual((x['count'],x['frequency']),(2,1))
            self.assertLess(x['lower'],1);self.assertEqual(x['upper'],1)

    def test_version_and_provenance(self):
        self.assertEqual(self.prov['model'],'Qwen/Qwen3-8B')
        self.assertEqual(self.prov['as_of'],'2026-01')
        self.assertEqual(len(self.entries),2)
        with self.assertRaises(ValueError):m.read_input('example',None,3)

if __name__=='__main__':unittest.main()
