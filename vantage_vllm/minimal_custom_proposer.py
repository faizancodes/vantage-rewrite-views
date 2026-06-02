"""Minimal proposer class for vLLM ``custom_class`` API probes.

The class is deliberately permissive: it accepts arbitrary constructor
arguments and exposes the CPU no-model proposer methods used by vLLM's built-in
ngram proposer. It always proposes zero draft tokens, so a successful smoke run
only proves that vLLM accepted and invoked a custom proposer path; it is not a
performance result.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class MinimalCustomProposer:
    """No-op custom proposer for API compatibility smoke tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.propose_calls = 0
        self._log_event("init", {"args_count": len(args), "kwargs": sorted(kwargs)})

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        self._log_event("load_model", {"args_count": len(args), "kwargs": sorted(kwargs)})
        return None

    def propose(self, *args: Any, **kwargs: Any) -> list[list[int]]:
        self.propose_calls += 1
        num_sequences = _infer_num_sequences(args, kwargs)
        self._log_event(
            "propose",
            {
                "args_count": len(args),
                "kwargs": sorted(kwargs),
                "num_sequences": num_sequences,
                "propose_calls": self.propose_calls,
            },
        )
        return [[] for _ in range(num_sequences)]

    def __call__(self, *args: Any, **kwargs: Any) -> list[list[int]]:
        self._log_event("__call__", {"args_count": len(args), "kwargs": sorted(kwargs)})
        return self.propose(*args, **kwargs)

    def _log_event(self, event: str, payload: dict[str, Any]) -> None:
        path = os.environ.get("VANTAGE_VLLM_MINIMAL_PROPOSER_LOG")
        if not path:
            return
        row = {"event": event, "timestamp": time.time(), **payload}
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _infer_num_sequences(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    if "num_tokens_no_spec" in kwargs:
        try:
            return len(kwargs["num_tokens_no_spec"])
        except TypeError:
            pass
    if len(args) >= 2:
        try:
            return len(args[1])
        except TypeError:
            pass
    if "token_ids_cpu" in kwargs:
        shape = getattr(kwargs["token_ids_cpu"], "shape", None)
        if shape:
            return int(shape[0])
    if len(args) >= 3:
        shape = getattr(args[2], "shape", None)
        if shape:
            return int(shape[0])
    return 1
