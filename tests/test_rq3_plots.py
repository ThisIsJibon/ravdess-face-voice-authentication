"""Regression coverage for the complete RQ3 severity grid and its summaries."""
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mbproject.common import ROOT
from mbproject.experiment import SYSTEMS
from mbproject.plots import rq3_curves


def study(sigmas):
    config = {
        "audio_snr_db": [None, 20, 10, 0],
        "face_blur_sigma": sigmas,
        "degradation_enrollment_sizes": [1, 4, 16],
    }
    rows = []
    for k in config["degradation_enrollment_sizes"]:
        for snr_index, snr in enumerate(["clean", "20", "10", "0"]):
            for sigma in sigmas:
                for system_index, system in enumerate(SYSTEMS):
                    for seed_index, seed in enumerate([73, 101]):
                        error = .003 * sigma + .002 * snr_index + .001 * system_index + .0001 * seed_index
                        rows.append(dict(k_face=k, k_voice=k, audio_snr_db=snr, face_sigma=sigma,
                                         system=system, seed=seed, eer=error, hw3_eer=error + .0005,
                                         dprime=10 - sigma / 2,
                                         **{"frr_at_dev_far_0.01": error * 2,
                                            "far_at_dev_far_0.01": .001 + .0001 * seed_index}))
    return config, pd.DataFrame(rows)


@pytest.mark.parametrize("sigmas", [[0, 1, 2, 4], [0, 1, 2, 4, 6, 8]])
def test_joint_plot_contains_every_blur_at_fixed_noise_and_summary_preserves_units(tmp_path, sigmas):
    config, metrics = study(sigmas)
    seen = []

    def save(fig, name):
        k = int(name.rsplit("k", 1)[1])
        seen.append(k)
        joint = fig.axes[2]
        assert "0 dB" in joint.get_title()
        assert [tick.get_text() for tick in joint.get_xticklabels()] == [f"sigma {s}" for s in sigmas]
        assert "Interpolated EER" in fig.axes[0].get_ylabel()
        for system, line in zip(SYSTEMS, joint.lines):
            expected = metrics[(metrics.k_face == k) & (metrics.system == system)
                               & (metrics.audio_snr_db == "0")].groupby("face_sigma").eer.mean()
            np.testing.assert_allclose(line.get_ydata(), expected.reindex(sigmas).to_numpy() * 100)
            assert len(line.get_xdata()) == len(sigmas)
        plt.close(fig)

    rq3_curves(config, metrics, tmp_path, save)
    assert seen == [1, 4, 16]
    table = pd.read_csv(tmp_path / "rq3_summary.csv")
    assert len(table) == 3 * 4 * len(sigmas) * len(SYSTEMS)
    assert {"eer_mean", "hw3_eer_mean", "dprime_mean", "far_at_dev_far_0.01_mean"} <= set(table.columns)
    actual = table[(table.k == 16) & (table.audio_snr_db == "0") & (table.face_sigma == sigmas[-1])
                   & (table.system == "face")].iloc[0]
    expected = metrics[(metrics.k_face == 16) & (metrics.audio_snr_db == "0")
                       & (metrics.face_sigma == sigmas[-1]) & (metrics.system == "face")]
    assert actual.eer_mean == pytest.approx(expected.eer.mean())
    assert actual.hw3_eer_mean == pytest.approx(expected.hw3_eer.mean())
    assert actual["far_at_dev_far_0.01_mean"] == pytest.approx(expected["far_at_dev_far_0.01"].mean())


def test_rate_bands_stay_within_zero_and_one_hundred_percent(tmp_path):
    config, metrics = study([0, 1, 2, 4, 6, 8])
    metrics["frr_at_dev_far_0.01"] = np.where(metrics.seed == 73, .6, 1.0)

    def save(fig, name):
        for ax in fig.axes[3:]:
            for band in ax.collections:
                y = band.get_paths()[0].vertices[:, 1]
                assert np.min(y) >= 0
                assert np.max(y) <= 100
        plt.close(fig)

    rq3_curves(config, metrics, tmp_path, save)


def test_missing_strongest_joint_condition_fails_instead_of_omitting_it(tmp_path):
    config, metrics = study([0, 1, 2, 4, 6, 8])
    metrics = metrics[~((metrics.system == "face") & (metrics.audio_snr_db == "0") & (metrics.face_sigma == 8))]
    try:
        with pytest.raises(ValueError, match="Missing RQ3 rows"):
            rq3_curves(config, metrics, tmp_path, lambda fig, name: plt.close(fig))
    finally:
        plt.close("all")


def test_notebook_module_backend_falls_back_before_matplotlib_import():
    env = {**os.environ, "MPLBACKEND": "module://unavailable_notebook_backend", "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", "import mbproject, matplotlib; assert matplotlib.get_backend().lower() == 'agg'"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
