import json
import subprocess
import sys

import pytest

from asts.diff_hunk_generation import PatchError, apply_patch_text, evaluate_completion


def test_applies_unified_diff_hunk():
    source = "a = 1\nb = 2\nc = 3\n"
    patch = """--- a/example.py
+++ b/example.py
@@ -1,3 +1,3 @@
 a = 1
-b = 2
+b = 20
 c = 3
"""

    assert apply_patch_text(source, patch) == "a = 1\nb = 20\nc = 3\n"


def test_rejects_unified_diff_context_mismatch_cleanly():
    source = "a = 1\nb = 2\n"
    patch = """@@ -1,2 +1,2 @@
 a = 1
-missing = 2
+b = 3
"""

    with pytest.raises(PatchError) as exc:
        apply_patch_text(source, patch)

    assert exc.value.code == "hunk_context_mismatch"


def test_applies_unified_diff_insertion_hunk():
    source = "a = 1\n"
    patch = """@@ -1,0 +2,1 @@
+b = 2
"""

    assert apply_patch_text(source, patch) == "a = 1\nb = 2\n"


def test_applies_json_anchor_replacements():
    source = "def f():\n    return old_name(value)\n"
    patch = json.dumps(
        {
            "replacements": [
                {
                    "start_anchor": "    return ",
                    "end_anchor": "(value)",
                    "replacement": "new_name",
                }
            ]
        }
    )

    assert apply_patch_text(source, patch) == "def f():\n    return new_name(value)\n"


def test_rejects_ambiguous_json_anchor():
    source = "start old end\nstart old end\n"
    patch = json.dumps(
        {
            "start_anchor": "start ",
            "end_anchor": " end",
            "replacement": "new",
        }
    )

    with pytest.raises(PatchError) as exc:
        apply_patch_text(source, patch)

    assert exc.value.code == "ambiguous_start_anchor"


def test_applies_search_replace_hunk():
    source = "alpha()\nbeta()\n"
    patch = """<<<<<<< SEARCH
beta()
=======
gamma()
>>>>>>> REPLACE
"""

    assert apply_patch_text(source, patch) == "alpha()\ngamma()\n"


def test_rejects_ambiguous_search_replace_hunk():
    source = "beta()\nbeta()\n"
    patch = """<<<<<<< SEARCH
beta()
=======
gamma()
>>>>>>> REPLACE
"""

    with pytest.raises(PatchError) as exc:
        apply_patch_text(source, patch)

    assert exc.value.code == "ambiguous_search"


def test_evaluate_completion_reports_metrics():
    source = "a = 1\n"
    expected = "a = 2\n"
    patch = """@@ -1,1 +1,1 @@
-a = 1
+a = 2
"""

    row = evaluate_completion(source, patch, expected=expected)

    assert row["parse_success"] is True
    assert row["apply_success"] is True
    assert row["exact_match"] is True
    assert row["edit_distance"] == 0
    assert row["output_length"] == len(expected)


def test_cli_writes_reports_for_jsonl_dataset(tmp_path):
    input_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "out"
    input_path.write_text(
        json.dumps(
            {
                "task_id": "one",
                "source": "x = 1\n",
                "expected": "x = 2\n",
                "completion": "@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n",
            }
        )
        + "\n"
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_diff_hunk_generation_eval.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    report = json.loads((output_dir / "report.json").read_text())
    assert (output_dir / "report.md").exists()
    assert report["groups"][0]["apply_success"] == 1
    assert report["groups"][0]["exact_match"] == 1
