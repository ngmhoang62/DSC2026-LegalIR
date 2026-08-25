"""EXP-019 overlap screen using the reusable E5-safe EXP-018 evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import exp018_e5_chunk_config_screen as screen

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "exp019_e5_overlap"
CORPORA = {
    "e5_508_o0": ROOT / "cache" / "exp019_e5_508_o0",
    "e5_508_o32": ROOT / "cache" / "exp018_e5_508_o32",
    "e5_508_o64": ROOT / "cache" / "exp019_e5_508_o64",
}


def summarize() -> Path:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in RESULTS.glob("e5_508_o*.json")]
    reused_o32 = ROOT / "results" / "exp018_e5_chunk_config" / "e5_508_o32.json"
    if reused_o32.exists() and not any(row["corpus_key"] == "e5_508_o32" for row in reports):
        reports.append(json.loads(reused_o32.read_text(encoding="utf-8")))
    reports.sort(key=lambda row: row["chunk_config"]["token_overlap"])
    lines = ["# EXP-019 E5-508 overlap screen", "", "> Fold-0 development screen only. Parser, tokenizer, size and fixture are fixed.", "", "| Overlap | Full config | Fixture chunks | E5 input max | R@5 | R@20 | R@100 | MRR@5 | sec |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in reports:
        cfg, m = row["chunk_config"], row["metrics"]
        lines.append(f"| {cfg['token_overlap']} | {cfg['max_passage_tokens']}/{cfg['token_window']}/{cfg['token_overlap']} | {row['fixture']['chunks']} | {row['e5_input_tokens']['max']} | {m['recall_at_5']:.4f} | {m['recall_at_20']:.4f} | {m['recall_at_100']:.4f} | {m['mrr_at_5']:.4f} | {row['runtime']['elapsed_seconds']:.1f} |")
    output = RESULTS / "summary.md"
    output.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run-one", "summarize"), required=True)
    parser.add_argument("--config", choices=tuple(CORPORA))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    screen.CORPORA = CORPORA
    screen.RESULTS = RESULTS
    if args.stage == "run-one":
        if not args.config: parser.error("--config is required")
        print(json.dumps(screen.run(args.config,args.device,args.batch_size,args.force),ensure_ascii=False,indent=2))
    else: print(summarize())


if __name__ == "__main__": main()
