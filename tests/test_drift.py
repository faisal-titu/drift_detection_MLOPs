"""Tests for drift detection module."""

import numpy as np
import pandas as pd
import pytest

from drift.drift_check import (
    calculate_psi,
    ks_drift_check,
    psi_drift_check,
    check_drift,
    load_reference_data,
)
from drift.simulate_drift import (
    generate_no_drift,
    generate_mild_drift,
    generate_heavy_drift,
)


@pytest.fixture
def reference_data():
    """Load reference data fixture."""
    return load_reference_data()


class TestPSI:
    """Test PSI calculation."""

    def test_psi_same_distribution(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = calculate_psi(data, data)
        assert psi < 0.1, f"PSI should be low for same distribution, got {psi}"

    def test_psi_different_distribution(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        drifted = np.random.normal(3, 1, 1000)
        psi = calculate_psi(ref, drifted)
        assert psi > 0.2, f"PSI should be high for different distributions, got {psi}"


class TestKSTest:
    """Test KS-test drift detection."""

    def test_no_drift(self, reference_data):
        sample = reference_data.sample(n=500, random_state=42)
        results = ks_drift_check(reference_data, sample)
        
        drifted = [f for f, r in results.items() if r["drift_detected"]]
        assert len(drifted) <= 1, f"Too many false positives: {drifted}"

    def test_heavy_drift(self, reference_data):
        drifted_data = generate_heavy_drift(reference_data, n_samples=500)
        results = ks_drift_check(reference_data, drifted_data)
        
        drifted = [f for f, r in results.items() if r["drift_detected"]]
        assert len(drifted) >= 3, f"Should detect drift in many features, got {drifted}"


class TestDriftCheck:
    """Test combined drift check."""

    def test_no_drift_detected(self, reference_data):
        no_drift = generate_no_drift(reference_data, n_samples=500)
        detected, report = check_drift(no_drift, reference_data)
        assert detected is False

    def test_heavy_drift_detected(self, reference_data):
        heavy = generate_heavy_drift(reference_data, n_samples=500)
        detected, report = check_drift(heavy, reference_data)
        assert detected is True
        assert report["drift_detected"] is True
