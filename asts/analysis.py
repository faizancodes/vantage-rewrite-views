"""Decision-gate analysis: combine tree-sitter and model latencies into a
projected end-to-end speedup, then print a clear PROCEED / PIVOT / KILL verdict.

Speed model
-----------

Vanilla autoregressive (no spec): per outer step, generate 1 token.
    cost_AR = target_ar_step

Spec-decode with tree-sitter gating (ASTS-Spec, lossless): per outer step,
draft proposes k tokens; the *gate* runs incremental tree-sitter parse on the
draft suffix to validate (cheap pre-filter); target verifies all k in one
forward pass; on average `a` tokens are accepted (a ≤ k).

    cost_SPEC = k * draft_ar_step              # draft generates k tokens
              + a * parse_us_inc_1tok          # parse cost amortized per accept
              + target_verify_kstep            # target verifies the draft

    tokens_generated_per_step = a + 1   # +1 for the bonus token from target

    speedup = (cost_AR * tokens_generated_per_step) / cost_SPEC

We sweep `a` over plausible accepted-length values [1, 2, 4, 6, 8] and
report the speedup curve. We also separately report the "pure parse
overhead" — the parse cost as a fraction of the savings — so we know whether
parsing is the bottleneck.

Notes
-----
- We use INCREMENTAL parse cost, not cold parse. Cold parse only happens
  once per request (initial prefill). Per token thereafter, tree-sitter
  reuses the prior tree.
- For verify_kstep parse, we use the k-step incremental parse number — that's
  the cost of validating the whole draft subtree in one parse call, which
  is what the gate would actually do.
- We compute speedup PER LANGUAGE since parse cost differs (TypeScript
  grammar is bigger than Python's).
"""

from __future__ import annotations

import statistics
from typing import Iterable


def _filter(measurements: list[dict], **kw) -> list[dict]:
    out = []
    for m in measurements:
        if all(m.get(k) == v for k, v in kw.items()):
            out.append(m)
    return out


def _avg_p50(measurements: list[dict]) -> float | None:
    if not measurements:
        return None
    return statistics.fmean(m["stats_us"]["p50"] for m in measurements)


def compute_verdict(
    treesitter_results: dict,
    model_results: dict,
    a_values: Iterable[int] = (1, 2, 4, 6, 8),
    k: int = 8,
) -> dict:
    """Compute the speedup verdict.

    Args:
        treesitter_results: output of asts.treesitter_bench.run_sweep(...)
        model_results: output of asts.model_bench.run_sweep(...)
        a_values: accepted-token-length values to sweep over
        k: speculative draft length used for the verify_kstep cost
    """
    target_id = model_results["target_id"]
    draft_id = model_results["draft_id"]
    measurements = model_results["measurements"]
    ts_measurements = treesitter_results["measurements"]

    # Find a representative prefix length used in the model bench.
    # We pick the LARGEST since long-context AR is where spec decode actually
    # shines; short-context AR is so fast no method helps much.
    prefix_lens = sorted({m["prefix_tokens"] for m in measurements if m["operation"] == "ar_step"})
    if not prefix_lens:
        return {"verdict": "ERROR", "reason": "no ar_step measurements found"}
    chosen_prefix = prefix_lens[-1]

    target_ar = _avg_p50(_filter(measurements, model_id=target_id, operation="ar_step", prefix_tokens=chosen_prefix))
    draft_ar = _avg_p50(_filter(measurements, model_id=draft_id, operation="ar_step", prefix_tokens=chosen_prefix))
    target_verify_k = _avg_p50(
        _filter(measurements, model_id=target_id, operation="verify_kstep", prefix_tokens=chosen_prefix, k=k)
    )

    if target_ar is None or draft_ar is None or target_verify_k is None:
        return {"verdict": "ERROR", "reason": f"missing model measurements for k={k} prefix={chosen_prefix}"}

    by_language: dict = {}

    for lang in ("python", "typescript"):
        # Per-token incremental parse cost (used per accepted token)
        parse_inc_1tok = _avg_p50(_filter(ts_measurements, language=lang, operation="incremental_1tok"))
        # k-step parse cost (used per outer spec step to validate the draft subtree)
        parse_kstep = _avg_p50(_filter(ts_measurements, language=lang, operation="incremental_kstep", k=k))
        # Cold parse for reference / prefill
        parse_cold_p50 = _avg_p50(_filter(ts_measurements, language=lang, operation="cold"))

        if parse_inc_1tok is None or parse_kstep is None:
            by_language[lang] = {"error": f"missing tree-sitter measurements for k={k}"}
            continue

        speedups: list[dict] = []
        for a in a_values:
            # Spec cost per outer step:
            cost_spec = (
                k * draft_ar  # draft generates k tokens
                + parse_kstep  # validate the draft subtree (one parse call)
                + target_verify_k  # target verifies k draft tokens in one fwd
            )
            tokens_per_step = a + 1
            tokens_per_us_spec = tokens_per_step / cost_spec
            tokens_per_us_ar = 1.0 / target_ar
            speedup = tokens_per_us_spec / tokens_per_us_ar

            # Parse overhead: parse cost as fraction of savings vs vanilla AR.
            ar_baseline_for_step = tokens_per_step * target_ar
            savings = ar_baseline_for_step - cost_spec
            parse_cost_in_step = parse_kstep
            parse_overhead_pct = (
                100.0 * parse_cost_in_step / max(savings, 1.0)
                if savings > 0
                else float("inf")
            )

            speedups.append({
                "accepted_tokens_a": a,
                "tokens_per_outer_step": tokens_per_step,
                "cost_spec_us": cost_spec,
                "cost_ar_equivalent_us": ar_baseline_for_step,
                "speedup_x": speedup,
                "parse_overhead_pct_of_savings": parse_overhead_pct,
                "savings_us": savings,
            })

        by_language[lang] = {
            "parse_us": {
                "cold_p50": parse_cold_p50,
                "incremental_1tok_p50": parse_inc_1tok,
                f"incremental_kstep_k{k}_p50": parse_kstep,
            },
            "speedup_curve": speedups,
        }

    # ---- Overall verdict --------------------------------------------------
    # Decision rule: at typical accepted length a=4, with k=8:
    #   speedup >= 1.5x AND parse overhead <= 30% → PROCEED
    #   speedup >= 1.0x AND parse overhead <= 60% → CAUTIOUS PROCEED
    #   otherwise                                 → PIVOT
    #   speedup <  0.9x at any a                  → KILL
    typical_a = 4

    verdict_per_lang: dict = {}
    for lang, d in by_language.items():
        if "error" in d:
            verdict_per_lang[lang] = "MISSING"
            continue
        row = next((s for s in d["speedup_curve"] if s["accepted_tokens_a"] == typical_a), None)
        if row is None:
            verdict_per_lang[lang] = "MISSING"
            continue
        sp = row["speedup_x"]
        ov = row["parse_overhead_pct_of_savings"]
        if sp >= 1.5 and ov <= 30.0:
            v = "PROCEED"
        elif sp >= 1.0 and ov <= 60.0:
            v = "CAUTIOUS_PROCEED"
        elif sp < 0.9:
            v = "KILL"
        else:
            v = "PIVOT"
        verdict_per_lang[lang] = v

    # Overall: take the worst (most pessimistic) verdict across the languages
    # we plan to support. Both must pass for the cross-language story.
    order = ["KILL", "PIVOT", "CAUTIOUS_PROCEED", "PROCEED", "MISSING"]
    worst = "PROCEED"
    for v in verdict_per_lang.values():
        if order.index(v) < order.index(worst):
            worst = v

    return {
        "schema": "asts-spec/verdict/v1",
        "config": {
            "target_id": target_id,
            "draft_id": draft_id,
            "prefix_tokens": chosen_prefix,
            "k": k,
            "a_values": list(a_values),
            "typical_a": typical_a,
        },
        "model_us": {
            "target_ar_step_p50": target_ar,
            "draft_ar_step_p50": draft_ar,
            f"target_verify_k{k}_p50": target_verify_k,
        },
        "by_language": by_language,
        "verdict_per_language": verdict_per_lang,
        "verdict": worst,
        "decision_rule": {
            "PROCEED": "speedup>=1.5x at a=4 AND parse_overhead<=30%",
            "CAUTIOUS_PROCEED": "speedup>=1.0x at a=4 AND parse_overhead<=60%",
            "KILL": "speedup<0.9x at a=4",
            "PIVOT": "everything else",
        },
    }


# ---------------------------------------------------------------------------
# Pretty-printer for the entrypoint
# ---------------------------------------------------------------------------


def print_verdict(verdict: dict) -> None:
    print()
    print("=" * 78)
    print("ASTS-Spec Decision Gate: Microbenchmark Verdict")
    print("=" * 78)

    cfg = verdict["config"]
    print()
    print(f"  target model:      {cfg['target_id']}")
    print(f"  draft model:       {cfg['draft_id']}")
    print(f"  prefix tokens:     {cfg['prefix_tokens']}")
    print(f"  draft length k:    {cfg['k']}")

    mu = verdict["model_us"]
    print()
    print("  Model latency (p50, microseconds, on warmed cache):")
    print(f"    target ar_step:           {mu['target_ar_step_p50']:>9.1f} us")
    print(f"    draft ar_step:            {mu['draft_ar_step_p50']:>9.1f} us")
    key = f"target_verify_k{cfg['k']}_p50"
    print(f"    target verify (k={cfg['k']}):       {mu[key]:>9.1f} us")
    print()

    for lang, d in verdict["by_language"].items():
        print(f"  --- {lang.upper()} ---")
        if "error" in d:
            print(f"    {d['error']}")
            continue
        p = d["parse_us"]
        print(f"    parse cold p50:           {p['cold_p50']:>9.1f} us")
        print(f"    parse incremental 1tok:   {p['incremental_1tok_p50']:>9.1f} us")
        kkey = f"incremental_kstep_k{cfg['k']}_p50"
        print(f"    parse incremental k={cfg['k']}:   {p[kkey]:>9.1f} us")
        print()
        print(
            f"    {'a':>3}  {'tok/step':>8}  {'spec_us':>9}  {'ar_us':>9}  "
            f"{'speedup':>8}  {'parse_ovh%':>10}"
        )
        print("    " + "-" * 58)
        for s in d["speedup_curve"]:
            mark = ""
            if s["speedup_x"] >= 1.5:
                mark = " ✓"
            elif s["speedup_x"] >= 1.0:
                mark = " ~"
            else:
                mark = " ✗"
            ov = s["parse_overhead_pct_of_savings"]
            ov_str = f"{ov:>9.1f}%" if ov != float("inf") else "      inf"
            print(
                f"    {s['accepted_tokens_a']:>3}  "
                f"{s['tokens_per_outer_step']:>8}  "
                f"{s['cost_spec_us']:>9.1f}  "
                f"{s['cost_ar_equivalent_us']:>9.1f}  "
                f"{s['speedup_x']:>7.2f}x  "
                f"{ov_str}{mark}"
            )
        print(f"    verdict: {verdict['verdict_per_language'][lang]}")
        print()

    print("=" * 78)
    v = verdict["verdict"]
    if v == "PROCEED":
        print("  ✓ VERDICT: PROCEED")
        print("    Tree-sitter parse cost is well below the spec-decode budget.")
        print("    Build the full ASTS-Spec prototype.")
    elif v == "CAUTIOUS_PROCEED":
        print("  ~ VERDICT: CAUTIOUS PROCEED")
        print("    Speedup is achievable but parse overhead is non-trivial.")
        print("    Consider amortizing parse calls (every n-tokens, not per-token).")
    elif v == "PIVOT":
        print("  ⚠ VERDICT: PIVOT")
        print("    Marginal speedup. Either:")
        print("      (a) Replace tree-sitter w/ a lighter check (e.g. brace-balance).")
        print("      (b) Frame as quality-improvement (constrained gen) not speed.")
        print("      (c) Pick a slower target model where AR cost dominates parse.")
    elif v == "KILL":
        print("  ✗ VERDICT: KILL")
        print("    Tree-sitter is too expensive in the decode loop.")
        print("    Consider grammar-compiled mask (XGrammar) or training-time AST signals.")
    else:
        print(f"  ? VERDICT: {v} — see details above")
    print(f"  rule: {verdict['decision_rule'][v] if v in verdict['decision_rule'] else 'n/a'}")
    print("=" * 78)
    print()
