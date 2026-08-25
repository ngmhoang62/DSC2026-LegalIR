from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp031_model_specific_validation import BASELINE, label_blind_sample, run_job, select_policies


class ModelSpecificValidationTests(unittest.TestCase):
    def test_label_blind_sample_is_deterministic_and_outer_specific(self) -> None:
        qids = [str(index) for index in range(100)]
        first = label_blind_sample(qids, outer="fold_0", limit=10)
        self.assertEqual(first, label_blind_sample(list(reversed(qids)), outer="fold_0", limit=10))
        self.assertNotEqual(first, label_blind_sample(qids, outer="fold_1", limit=10))
        self.assertEqual(len(first), 10)

    def test_selector_is_outer_and_model_isolated_with_safe_baseline(self) -> None:
        rows = []
        for outer in ("fold_0", "fold_1"):
            inner = "fold_1" if outer == "fold_0" else "fold_0"
            for model in ("bge_m3", "gte"):
                for variant in ("evidence_only", "title_base", "both_base", "typed_scope", "multi_view"):
                    delta = 0.01 if (outer, model, variant) == ("fold_0", "bge_m3", "multi_view") else -0.01
                    rows.append({
                        "outer": outer, "inner": inner, "model": model, "variant": variant,
                        "delta_recall@5": delta,
                        "baseline": {"precision@5": 0.1}, "metrics": {"precision@5": 0.11 if delta > 0 else 0.09},
                    })
        policies = select_policies({"rows": rows})
        self.assertEqual(policies["fold_0"]["bge_m3"]["variant"], "multi_view")
        self.assertEqual(policies["fold_1"]["bge_m3"]["variant"], BASELINE)
        self.assertEqual(policies["fold_0"]["gte"]["variant"], BASELINE)

    def test_resume_state_is_not_overwritten_by_success_marker(self) -> None:
        from tempfile import TemporaryDirectory
        import json
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            (output / "scores.jsonl").write_text("{}\n", encoding="utf-8")
            marker = root / "jobs" / "job" / "_SUCCESS.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({"fingerprint": "same", "state": "SUCCESS"}), encoding="utf-8")
            result = run_job(
                root=root, name="job", fingerprint="same", output=output, model="gte",
                capsules=root / "unused.jsonl", qids={"1"}, variant=BASELINE,
                device="cpu", local_only=True, resume=True,
            )
            self.assertEqual(result["state"], "SKIPPED_RESUME")


if __name__ == "__main__":
    unittest.main()
