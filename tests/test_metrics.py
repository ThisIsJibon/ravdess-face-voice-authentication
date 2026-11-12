import numpy as np
import pytest

from mbproject.metrics import (FastEvaluator, eer, evaluate, fit_policies,
                               rates_at_threshold, threshold_at_far)
from mbproject.vendor.hw3_evaluator import Evaluator


def test_perfect_separation_has_zero_eer_and_unit_auc():
    scores = np.array([.9,.8,.2,.1])
    labels = np.array([True,True,False,False])
    policies = fit_policies(scores,labels,[.01,.001,.0001])
    result = evaluate(scores,labels,policies,[.01,.001,.0001])
    assert result["eer"] == 0
    assert result["auc"] == 1
    assert result["frr_at_dev_far_0.01"] == 0
    assert result["far_at_dev_far_0.01"] == 0


def test_all_ties_are_chance_not_perfect():
    scores = np.full(8,.5)
    labels = np.array([True]*4+[False]*4)
    assert eer(scores,labels) == pytest.approx(.5)
    threshold = threshold_at_far(scores,labels,.01)
    assert threshold > .5
    assert rates_at_threshold(scores,labels,threshold) == dict(far=0,frr=1,false_accepts=0,false_rejects=4)


@pytest.mark.parametrize("target",[0,.01,.1,.25,.5,1])
def test_far_constraint_handles_impostor_ties_conservatively(target):
    scores = np.array([.95,.8,.75,.5,.8,.8,.4,.1])
    labels = np.array([True]*4+[False]*4)
    threshold = threshold_at_far(scores,labels,target)
    assert rates_at_threshold(scores,labels,threshold)["far"] <= target


def test_accelerated_hw3_counts_match_submitted_script():
    rng = np.random.default_rng(73)
    genuine = np.r_[rng.uniform(size=71),.5,.5,0,1]
    impostor = np.r_[rng.uniform(size=117),.5,.5,0,1]
    slow = Evaluator(500,genuine,impostor,"parity")
    fast = FastEvaluator(500,genuine,impostor,"parity")
    for expected,actual in zip(slow.compute_rates(),fast.compute_rates()):
        np.testing.assert_allclose(expected,actual,rtol=0,atol=0)
    assert slow.get_dprime() == fast.get_dprime()
    assert slow.get_EER(*slow.compute_rates()[:2]) == fast.get_EER(*fast.compute_rates()[:2])


def test_operational_threshold_does_not_refit_on_degraded_test_scores():
    dev = np.array([.9,.85,.4,.3])
    labels = np.array([True,True,False,False])
    policies = fit_policies(dev,labels,[.01])
    test = np.array([.5,.45,.6,.55])
    result = evaluate(test,labels,policies,[.01])
    assert result["threshold_dev_far_0.01"] == policies["far_0.01"]
    assert result["frr_at_dev_far_0.01"] == 1


def test_dprime_uses_actual_hw3_formula():
    genuine,impostor = np.array([.6,.8,1.]),np.array([.1,.2,.3])
    expected = np.sqrt(2)*abs(genuine.mean()-impostor.mean())/(np.sqrt(genuine.var()+impostor.var())+1e-12)
    assert FastEvaluator(500,genuine,impostor,"dprime").get_dprime() == pytest.approx(expected)


@pytest.mark.parametrize("scores,labels",[([np.nan,.1],[True,False]),([.5,.1],[True,True])])
def test_invalid_trials_fail(scores,labels):
    with pytest.raises(ValueError):
        eer(scores,labels)
