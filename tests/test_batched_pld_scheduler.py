import torch

from scripts.run_batched_pld_eval import (
    ActiveTask,
    _compact_cache_row,
    _combine_task_caches,
    _eos_truncate,
    _parse_ints,
    _to_legacy_cache,
    resolve_bucket_sizes,
)


def _cache_row(values):
    t = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1)
    return ((t, t + 1000),)


def test_parse_ints():
    assert _parse_ints("1,2, 4") == [1, 2, 4]


def test_resolve_bucket_sizes_final_policies():
    assert resolve_bucket_sizes("default", "") == [8, 16, 32, 64, 128]
    assert resolve_bucket_sizes("fine", "") == [1, 2, 4, 8, 16, 32, 64, 128]
    assert resolve_bucket_sizes("single", "") == [128]
    assert resolve_bucket_sizes("custom", "3,9") == [3, 9]


def test_eos_truncate_caps_at_first_eos_and_budget():
    assert _eos_truncate([1, 2, 3, 4], [3], 10) == ([1, 2, 3], True)
    assert _eos_truncate([1, 2, 3, 4], [9], 2) == ([1, 2], False)


def test_compact_cache_row_removes_padding_gap():
    key = torch.arange(10, dtype=torch.float32).reshape(2, 1, 5, 1)
    val = key + 100
    cache = ((key, val),)
    compacted = _compact_cache_row(
        cache,
        row=1,
        real_cache_len=2,
        max_cache_len=3,
        keep_input_len=2,
    )
    out_key = compacted[0][0].flatten().tolist()
    out_val = compacted[0][1].flatten().tolist()
    assert out_key == [5.0, 6.0, 8.0, 9.0]
    assert out_val == [105.0, 106.0, 108.0, 109.0]


def test_combine_task_caches_pads_task_local_rows():
    t1 = ActiveTask("a", "", [], [], 0, target_cache=_cache_row([1, 2]), target_cache_len=2)
    t2 = ActiveTask("b", "", [], [], 0, target_cache=_cache_row([3, 4, 5]), target_cache_len=3)
    combined, max_len = _combine_task_caches([t1, t2])
    legacy = _to_legacy_cache(combined)
    key = legacy[0][0]
    val = legacy[0][1]
    assert max_len == 3
    assert key.shape == (2, 1, 3, 1)
    assert key[0, 0, :, 0].tolist() == [1.0, 2.0, 0.0]
    assert key[1, 0, :, 0].tolist() == [3.0, 4.0, 5.0]
    assert val[0, 0, :, 0].tolist() == [1001.0, 1002.0, 0.0]
    assert val[1, 0, :, 0].tolist() == [1003.0, 1004.0, 1005.0]


def test_lagging_cache_invariant_for_partial_rejection():
    # Old cache covers two prefix tokens. Verifier input has one uncached
    # prefix anchor plus three draft tokens. If one draft accepts and the next
    # token is a correction from logits, only the anchor and accepted draft are
    # retained; the correction token is emitted but cached on the next step.
    key = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1)
    val = key + 100
    compacted = _compact_cache_row(
        ((key, val),),
        row=0,
        real_cache_len=2,
        max_cache_len=2,
        keep_input_len=2,  # n_pre=1 + accepted_drafts=1
    )
    assert compacted[0][0].flatten().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_lagging_cache_invariant_for_full_accept_with_bonus():
    # Full acceptance of three draft tokens emits a bonus token, but the bonus
    # KV is not present because the bonus was predicted, not fed as input.
    key = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1)
    val = key + 100
    compacted = _compact_cache_row(
        ((key, val),),
        row=0,
        real_cache_len=2,
        max_cache_len=2,
        keep_input_len=4,  # n_pre=1 + accepted_drafts=3
    )
    assert compacted[0][0].flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
