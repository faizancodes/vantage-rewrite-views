import subprocess
import sys

import torch

from asts.mtp_heads import MTPHeadConfig, PLDMTPHeads, accepted_prefix_length
from scripts.collect_queued_mtp_training_data import _eligible_pairs
from scripts.collect_router_selected_mtp_training_data import (
    _eligible_pairs as _router_selected_eligible_pairs,
)
from scripts.collect_pld_mtp_training_data import _hidden_and_label_positions
from scripts.evaluate_pld_mtp_heads_offline import MTPReplayStep, _project_policy, _trigger
from scripts.evaluate_queued_mtp_confidence import PredictionRecord, replay_confidence_threshold
from scripts.evaluate_queued_mtp_oracle import QueuedReplayStep, replay_perfect_queued_oracle
from scripts.evaluate_queued_mtp_with_weak_router import replay_with_router
from scripts.train_weak_pld_router import build_examples, split_tasks


def test_mtp_accepted_prefix_length() -> None:
    assert accepted_prefix_length([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert accepted_prefix_length([1, 2], [9, 2]) == 0
    assert accepted_prefix_length([1, 2], [1, 2, 3]) == 2


def test_mtp_trigger_policies() -> None:
    step = MTPReplayStep("task", 0, start=0, emitted=3, accepted_len=2, pld_miss=False)
    assert not _trigger("accepted_len_le_1", step)
    assert _trigger("accepted_len_le_2", step)
    assert not _trigger("pld_miss_only", step)


def test_mtp_replay_skips_future_steps() -> None:
    steps = {
        "task": [
            MTPReplayStep("task", 0, start=0, emitted=1, accepted_len=0, pld_miss=True),
            MTPReplayStep("task", 1, start=1, emitted=1, accepted_len=0, pld_miss=True),
            MTPReplayStep("task", 2, start=2, emitted=1, accepted_len=0, pld_miss=True),
            MTPReplayStep("task", 3, start=3, emitted=1, accepted_len=0, pld_miss=True),
        ]
    }
    result = _project_policy(
        steps_by_task=steps,
        policy="accepted_len_eq_0",
        num_heads=4,
        predictions={("task", 0): 4},
        oracle_upper_bound=False,
    )
    assert result["projected_steps"] == 1
    assert result["skipped_baseline_steps"] == 3


def test_post_pld_positions_advance_by_accepted_prefix() -> None:
    hidden_pos, label_start = _hidden_and_label_positions(
        generated_start=10,
        accepted_len=3,
        mtp_position="post_pld",
    )
    assert hidden_pos == 12
    assert label_start == 13


def test_pre_pld_positions_match_original_collection() -> None:
    hidden_pos, label_start = _hidden_and_label_positions(
        generated_start=10,
        accepted_len=3,
        mtp_position="pre_pld",
    )
    assert hidden_pos == 9
    assert label_start == 10


def test_post_pld_replay_combines_pld_prefix_and_mtp_prefix() -> None:
    steps = {
        "task": [
            MTPReplayStep("task", 0, start=0, emitted=3, accepted_len=2, pld_miss=False),
            MTPReplayStep("task", 1, start=3, emitted=1, accepted_len=0, pld_miss=True),
            MTPReplayStep("task", 2, start=4, emitted=1, accepted_len=0, pld_miss=True),
            MTPReplayStep("task", 3, start=5, emitted=1, accepted_len=0, pld_miss=True),
        ]
    }
    result = _project_policy(
        steps_by_task=steps,
        policy="accepted_len_le_2",
        num_heads=4,
        predictions={("task", 0): 4},
        oracle_upper_bound=False,
        mtp_position="post_pld",
    )
    assert result["projected_steps"] == 1
    assert result["skipped_baseline_steps"] == 3
    assert result["avg_extra_accepted_mtp_tokens_per_trigger"] == 3


def test_post_pld_replay_does_not_double_count_baseline_correction_token() -> None:
    steps = {
        "task": [
            MTPReplayStep("task", 0, start=0, emitted=3, accepted_len=2, pld_miss=False),
            MTPReplayStep("task", 1, start=3, emitted=1, accepted_len=0, pld_miss=True),
        ]
    }
    result = _project_policy(
        steps_by_task=steps,
        policy="accepted_len_le_2",
        num_heads=4,
        predictions={("task", 0): 1},
        oracle_upper_bound=False,
        mtp_position="post_pld",
    )
    assert result["projected_steps"] == 2
    assert result["avg_extra_accepted_mtp_tokens_per_trigger"] == 0


def test_post_pld_replay_progress_grid() -> None:
    cases = [
        # accepted_len, baseline emitted, mtp prefix, expected selected progress
        (0, 1, 0, 1),
        (0, 1, 1, 1),
        (0, 1, 2, 2),
        (0, 1, 4, 4),
        (1, 2, 0, 2),
        (1, 2, 1, 2),
        (1, 2, 2, 3),
        (1, 2, 4, 5),
        (4, 5, 0, 5),
        (4, 5, 1, 5),
        (4, 5, 2, 6),
        (4, 5, 4, 8),
    ]
    for accepted, emitted, mtp_prefix, expected in cases:
        steps = {"task": [MTPReplayStep("task", 0, start=0, emitted=emitted, accepted_len=accepted, pld_miss=False)]}
        result = _project_policy(
            steps_by_task=steps,
            policy="accepted_len_le_4",
            num_heads=4,
            predictions={("task", 0): mtp_prefix},
            oracle_upper_bound=False,
            mtp_position="post_pld",
        )
        assert result["avg_selected_emit_per_trigger"] == expected


def test_k4_heads_produce_expected_logit_shapes() -> None:
    model = PLDMTPHeads(
        MTPHeadConfig(hidden_size=8, vocab_size=16, num_heads=4, head_type="linear")
    )
    logits = model(torch.randn(3, 8))
    assert len(logits) == 4
    assert all(tuple(item.shape) == (3, 16) for item in logits)

    tied = PLDMTPHeads(
        MTPHeadConfig(hidden_size=8, vocab_size=16, num_heads=4, head_type="linear"),
        output_weight=torch.randn(16, 8),
    )
    fused_logits = tied.forward_logits(torch.randn(3, 8))
    pred = tied.predict_token_tensor(torch.randn(3, 8))
    assert tuple(fused_logits.shape) == (3, 4, 16)
    assert tuple(pred.shape) == (3, 4)


def test_perfect_queued_oracle_has_zero_token0_rejection() -> None:
    steps = {
        "task": [
            QueuedReplayStep("task", 0, start=0, emitted=1, accepted_len=0, draft_len=0, pld_miss=True),
            QueuedReplayStep("task", 1, start=1, emitted=1, accepted_len=0, draft_len=0, pld_miss=True),
            QueuedReplayStep("task", 2, start=2, emitted=1, accepted_len=0, draft_len=0, pld_miss=True),
            QueuedReplayStep("task", 3, start=3, emitted=1, accepted_len=0, draft_len=0, pld_miss=True),
            QueuedReplayStep("task", 4, start=4, emitted=1, accepted_len=0, draft_len=0, pld_miss=True),
        ]
    }
    result = replay_perfect_queued_oracle(steps, num_heads=4, trigger_threshold=4)
    assert result["oracle_token0_reject_count"] == 0
    assert result["queue_used"] >= 1
    assert result["oracle_accepted_queued_tokens_per_used_draft"] == 3


def test_confidence_gating_drops_low_confidence_queue() -> None:
    steps = {
        "task": [
            QueuedReplayStep("task", 0, start=0, emitted=1, accepted_len=0, draft_len=0, pld_miss=True),
            QueuedReplayStep("task", 1, start=1, emitted=1, accepted_len=0, draft_len=0, pld_miss=True),
        ]
    }
    predictions = {
        ("task", 0): PredictionRecord(
            task_id="task",
            step_id=0,
            queued_predictions=(10, 11, 12),
            queued_labels=(10, 11, 12),
            confidence=0.2,
            margin=0.1,
            baseline_token_matched=True,
        )
    }
    low = replay_confidence_threshold(steps, predictions, threshold=0.1)
    high = replay_confidence_threshold(steps, predictions, threshold=0.5)
    assert low["queue_used_after_gating"] == 1
    assert high["queue_used_after_gating"] == 0
    assert high["gated_drop_count"] == 1


def test_queued_use_dataset_pairs_create_step_with_next_weak_use_step() -> None:
    rows = [
        {"step": 0, "_generated_start": 0, "n_emitted": 1, "n_accepted_drafts": 0, "k": 0},
        {"step": 1, "_generated_start": 1, "n_emitted": 1, "n_accepted_drafts": 0, "k": 0},
        {"step": 2, "_generated_start": 2, "n_emitted": 5, "n_accepted_drafts": 4, "k": 64},
    ]
    pairs, counts = _eligible_pairs(
        rows,
        threshold=4,
        weak_field="draft_len",
        include_dropped=False,
    )
    assert len(pairs) == 1
    assert int(pairs[0][0]["step"]) == 0
    assert int(pairs[0][1]["step"]) == 1
    assert counts["dropped_pld_strong"] == 1


def test_weak_router_features_do_not_include_current_accepted_label() -> None:
    rows = {
        "task": [
            {
                "task_id": "task",
                "step": 0,
                "k": 8,
                "proposal_tokens": 8,
                "proposal_kind": "blazedit_pld",
                "n_accepted_drafts": 0,
                "n_emitted": 1,
                "_generated_start": 0,
            },
            {
                "task_id": "task",
                "step": 1,
                "k": 64,
                "proposal_tokens": 64,
                "proposal_kind": "blazedit_pld",
                "n_accepted_drafts": 32,
                "n_emitted": 33,
                "_generated_start": 1,
            },
        ]
    }
    feats, labels, _ = build_examples(rows, threshold=4)
    assert labels == [1, 0]
    assert "accepted_len" not in feats[0]
    assert "n_accepted_drafts" not in feats[0]
    assert feats[1]["prev_accept_len_1"] == 0


def test_task_id_split_prevents_row_leakage() -> None:
    rows = {f"task/{i}": [{"task_id": f"task/{i}", "step": 0}] for i in range(10)}
    train, test = split_tasks(rows, test_fraction=0.3, seed=1)
    assert train
    assert test
    assert not (set(train) & set(test))


def test_weak_router_projection_skips_future_steps_deterministically() -> None:
    steps = {
        "task": [
            QueuedReplayStep("task", 0, start=0, emitted=1, accepted_len=0, draft_len=64, pld_miss=False),
            QueuedReplayStep("task", 1, start=1, emitted=1, accepted_len=0, draft_len=64, pld_miss=False),
            QueuedReplayStep("task", 2, start=2, emitted=1, accepted_len=0, draft_len=64, pld_miss=False),
            QueuedReplayStep("task", 3, start=3, emitted=1, accepted_len=0, draft_len=64, pld_miss=False),
            QueuedReplayStep("task", 4, start=4, emitted=1, accepted_len=0, draft_len=64, pld_miss=False),
        ]
    }
    probs = {("task", i): 1.0 for i in range(5)}
    result1 = replay_with_router(steps, probs, threshold=0.5)
    result2 = replay_with_router(steps, probs, threshold=0.5)
    assert result1 == result2
    assert result1["projected_steps"] < result1["baseline_steps"]


def test_router_selected_collector_pairs_create_and_next_use_step() -> None:
    rows = {
        "task": [
            {"task_id": "task", "step": 0, "_generated_start": 0, "n_emitted": 1, "n_accepted_drafts": 0},
            {"task_id": "task", "step": 1, "_generated_start": 1, "n_emitted": 1, "n_accepted_drafts": 0},
            {"task_id": "task", "step": 2, "_generated_start": 2, "n_emitted": 9, "n_accepted_drafts": 8},
        ]
    }
    probs = {("task", 1): 0.6, ("task", 2): 0.8}
    pairs, summary = _router_selected_eligible_pairs(
        rows,
        probs,
        router_threshold=0.5,
        collection_threshold=0.5,
        accepted_len_threshold=4,
    )
    assert len(pairs["task"]) == 2
    create, use, prob, selected, is_tp = pairs["task"][0]
    assert int(create["step"]) == 0
    assert int(use["step"]) == 1
    assert prob == 0.6
    assert selected is True
    assert is_tp is True
    assert summary["counts"]["router_true_positive"] == 1
    assert summary["counts"]["router_false_positive"] == 1


def test_training_script_runs_tiny_synthetic_batch(tmp_path) -> None:
    data_path = tmp_path / "train.pt"
    out_path = tmp_path / "heads.pt"
    torch.save(
        {
            "hidden": torch.randn(8, 6),
            "labels": torch.randint(0, 12, (8, 4)),
            "accepted_len": torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]),
            "pld_miss": torch.tensor([True, False, False, False, False, True, False, False]),
            "task_id": [f"task/{i}" for i in range(8)],
            "step_id": list(range(8)),
            "metadata": {"hidden_size": 6},
        },
        data_path,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/train_pld_mtp_heads.py",
            "--data",
            str(data_path),
            "--output",
            str(out_path),
            "--num-heads",
            "4",
            "--head-type",
            "linear",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--lr",
            "1e-3",
            "--device",
            "cpu",
            "--vocab-size",
            "12",
            "--head-loss-weights",
            "4,2,1,1",
            "--max-train-examples",
            "4",
            "--eval-train-subset",
        ],
        check=True,
    )
    assert out_path.exists()
