import json

from asts.code_proposers import _apply_word_map, _rewrite_pairs
from asts.humaneval import load_problems_from_jsonl, stop_texts_for_language


def test_manifest_loader_preserves_metadata(tmp_path):
    path = tmp_path / "problems.jsonl"
    row = {
        "task_id": "drift/1",
        "prompt": "Edit this.\n```python\nx = 1\n```\n",
        "reference": "x = 1\n",
        "deterministic_target": "x_updated = 1\n",
        "language": "repo_edit_rename_python",
        "drift_family": "rename",
        "rewrite_pairs": {"x": "x_updated"},
    }
    path.write_text(json.dumps(row) + "\n")

    problems = load_problems_from_jsonl(str(path))

    assert len(problems) == 1
    assert problems[0].task_id == "drift/1"
    assert problems[0].reference == "x = 1\n"
    assert problems[0].deterministic_target == "x_updated = 1\n"
    assert problems[0].metadata["drift_family"] == "rename"
    assert problems[0].metadata["rewrite_pairs"] == {"x": "x_updated"}


def test_manifest_stop_texts_are_disabled_for_full_edit_outputs():
    assert stop_texts_for_language("manifest") == ()
    assert stop_texts_for_language("codeeditor_translate") == ()
    assert stop_texts_for_language("codeeditor_polish") == ()


def test_rewrite_pairs_support_literals_and_dotted_names():
    prompt = "replace `old.client` with `new.client` and change 30 to 60"

    pairs = _rewrite_pairs(prompt)
    rewritten = _apply_word_map("old.client.chat(timeout=30)\n", pairs)

    assert pairs["old.client"] == "new.client"
    assert pairs["30"] == "60"
    assert rewritten == "new.client.chat(timeout=60)\n"
