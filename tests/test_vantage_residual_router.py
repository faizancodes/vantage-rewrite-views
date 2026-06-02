from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from asts.vantage_residual_router import (
    ResidualRouter,
    ResidualRouterConfig,
    binary_metrics,
    build_labeled_feature_matrix,
    feature_dict_from_row,
    load_router_payload,
    should_trigger_residual,
    summarize_trigger_policy,
)


def test_rule_trigger_router_exports_residual_policy():
    assert should_trigger_residual("accepted_len_le_2", accepted_len=2)
    assert not should_trigger_residual("accepted_len_le_2", accepted_len=3)
    assert should_trigger_residual("router", accepted_len=4)


def test_feature_extraction_and_labeled_matrix():
    rows = [
        {"accepted_len": 0, "pld_miss": True, "proposal_len": 8, "step_id": 1},
        {"accepted_len": 6, "pld_miss": False, "proposal_len": 32, "step_id": 2},
    ]
    features = feature_dict_from_row(rows[0])
    assert features["accepted_len_eq_0"] == 1.0
    X, y = build_labeled_feature_matrix(rows, target_policy="accepted_len_le_4")
    assert X.shape[0] == 2
    assert y.tolist() == [1.0, 0.0]


def test_router_forward_and_metrics():
    model = ResidualRouter(ResidualRouterConfig(feature_dim=3, model_type="linear"))
    probs = model.forward_proba(torch.randn(5, 3))
    assert probs.shape == (5,)
    metrics = binary_metrics(
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([1.0, 0.0, 0.0, 1.0]),
    )
    assert metrics["true_positive"] == 1.0
    assert metrics["false_positive"] == 1.0


def test_summarize_trigger_policy_precision_recall():
    rows = [
        {"accepted_len": 0},
        {"accepted_len": 2},
        {"accepted_len": 5},
    ]
    summary = summarize_trigger_policy(rows, "accepted_len_le_1")
    assert summary["trigger_count"] == 1
    assert summary["actual_weak_count"] == 2
    assert summary["precision"] == 1.0
    assert summary["recall"] == 0.5


def test_load_router_payload_derives_labels_for_feature_tensor(tmp_path):
    path = tmp_path / "router.pt"
    torch.save(
        {
            "features": torch.randn(3, 5),
            "accepted_len": torch.tensor([0, 2, 6]),
            "pld_miss": torch.tensor([True, False, False]),
        },
        path,
    )
    payload = load_router_payload(path, target_policy="accepted_len_le_2")
    assert payload["features"].shape == (3, 5)
    assert len(payload["feature_names"]) == 5
    assert payload["labels"].tolist() == [1.0, 1.0, 0.0]
