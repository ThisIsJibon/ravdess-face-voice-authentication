"""HW3 evaluator integration plus exact ROC and development-only policies."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve

from .vendor.hw3_evaluator import Evaluator


class FastEvaluator(Evaluator):
    """Only accelerate threshold counts; inherited HW3 definitions are unchanged."""
    def compute_rates(self):
        genuine, impostor = np.sort(self.genuine_scores), np.sort(self.impostor_scores)
        fn = np.searchsorted(genuine, self.thresholds, side="left")
        tn = np.searchsorted(impostor, self.thresholds, side="left")
        return ((len(impostor)-tn)/(len(impostor)+self.epsilon),
                fn/(len(genuine)+self.epsilon),
                (len(genuine)-fn)/(len(genuine)+self.epsilon))


def roc(scores, genuine):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    genuine = np.asarray(genuine, dtype=bool).ravel()
    if not np.isfinite(scores).all() or genuine.all() or not genuine.any():
        raise ValueError("Need finite scores and both genuine and impostor trials")
    fpr, tpr, thresholds = roc_curve(genuine, scores, drop_intermediate=False)
    return fpr, 1-tpr, thresholds


def eer(scores, genuine):
    fpr, fnr, thresholds = roc(scores, genuine)
    difference = fpr-fnr
    j = int(np.flatnonzero(difference >= 0)[0])
    if j == 0:
        return float((fpr[j]+fnr[j])/2)
    a = -difference[j-1] / (difference[j]-difference[j-1])
    return float(fpr[j-1]+a*(fpr[j]-fpr[j-1]))


def threshold_at_far(scores, genuine, target):
    fpr, fnr, thresholds = roc(scores, genuine)
    allowed = np.flatnonzero(fpr <= target)
    # Thresholds descend; the last allowed point minimizes FRR subject to FAR.
    index = int(allowed[-1])
    threshold = thresholds[index]
    if not np.isfinite(threshold):
        threshold = np.nextafter(np.max(scores), np.inf)
    return float(threshold)


def rates_at_threshold(scores, genuine, threshold):
    scores, genuine = np.asarray(scores), np.asarray(genuine,dtype=bool)
    fp = int(np.sum(scores[~genuine] >= threshold))
    fn = int(np.sum(scores[genuine] < threshold))
    return dict(far=fp/int((~genuine).sum()), frr=fn/int(genuine.sum()),
                false_accepts=fp, false_rejects=fn)


def fit_policies(scores, genuine, targets):
    fpr, fnr, thresholds = roc(scores, genuine)
    index = int(np.argmin(np.abs(fpr-fnr)))
    threshold = thresholds[index]
    if not np.isfinite(threshold):
        threshold = np.nextafter(np.max(scores),np.inf)
    result = {"eer": float(threshold)}
    result.update({f"far_{target:g}":threshold_at_far(scores,genuine,target) for target in targets})
    return result


def evaluate(scores, genuine, policies, targets):
    scores, genuine = np.asarray(scores).ravel(), np.asarray(genuine,dtype=bool).ravel()
    g, i = scores[genuine], scores[~genuine]
    hw3 = FastEvaluator(500,g,i,"experiment")
    hw3_fpr, hw3_fnr, _ = hw3.compute_rates()
    hw3_eer, hw3_threshold = hw3.get_EER(hw3_fpr,hw3_fnr)
    result = dict(eer=eer(scores,genuine), hw3_eer=float(hw3_eer),
                  hw3_eer_threshold=float(hw3_threshold),dprime=float(hw3.get_dprime()),
                  auc=float(roc_auc_score(genuine,scores)),n_genuine=len(g),n_impostor=len(i),
                  far_resolution=1/len(i),genuine_mean=float(g.mean()),impostor_mean=float(i.mean()))
    for name, threshold in policies.items():
        values = rates_at_threshold(scores,genuine,threshold)
        result.update({f"{key}_at_dev_{name}":value for key,value in values.items()})
        result[f"gar_at_dev_{name}"] = 1-values["frr"]
        result[f"threshold_dev_{name}"] = threshold
    for target in targets:
        # Descriptive test-ROC quantities; these thresholds are NEVER deployed.
        threshold = threshold_at_far(scores,genuine,target)
        values = rates_at_threshold(scores,genuine,threshold)
        result[f"oracle_test_frr_at_far_{target:g}"] = values["frr"]
        result[f"oracle_test_achieved_far_{target:g}"] = values["far"]
    # Binomial bound is descriptive only: biometric pair trials are dependent.
    result["zero_error_binomial_upper95_reference"] = float(beta.ppf(.95,1,len(i)))
    return result
