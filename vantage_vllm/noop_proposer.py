"""Minimal read-only proposer stub for vLLM API experiments.

This class is not wired into vLLM. It mirrors the CPU-side no-model proposer
shape used by vLLM 0.20.2's built-in n-gram proposer closely enough for local
signature and monkey-patch experiments, while always returning no draft tokens.
"""

from __future__ import annotations

from typing import Any


class NoopProposer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = args
        self.init_kwargs = kwargs

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        *args: Any,
        **kwargs: Any,
    ) -> list[list[int]]:
        return [[] for _ in sampled_token_ids]

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        return None
