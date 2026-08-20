# EXP-013b runbook

Run from the LegalIR root with `d:\Study\DSC2026\dsc_env\Scripts\python.exe`.

```powershell
python src/exp013b_pipeline.py --stage preflight
python src/exp013b_pipeline.py --stage build-query-memory --split train
python src/exp013b_pipeline.py --stage build-candidates --split train
python src/exp013b_pipeline.py --stage audit-candidates --split train
python src/exp013b_pipeline.py --stage train-preranker-oof
python src/exp013b_pipeline.py --stage audit-shortlist
python src/exp013b_pipeline.py --stage prepare-qwen --allow-download
python src/exp013b_pipeline.py --stage build-capsules --split train --resume
python src/exp013b_pipeline.py --stage benchmark-qwen --pairs 2048
python src/exp013b_pipeline.py --stage score-qwen-screen --fold 0 --resume
python src/exp013b_pipeline.py --stage fit-router
```

If the Qwen benchmark fails both 768 and 512 token gates, replace only the
prepared scorer and benchmark with the approved fallback, then repeat the
fold-0 screen using the same `--model` value:

```powershell
python src/exp013b_pipeline.py --stage prepare-qwen --model Alibaba-NLP/gte-multilingual-reranker-base --allow-download
python src/exp013b_pipeline.py --stage benchmark-qwen --model Alibaba-NLP/gte-multilingual-reranker-base --allow-download
python src/exp013b_pipeline.py --stage score-qwen-screen --fold 0 --model Alibaba-NLP/gte-multilingual-reranker-base --resume
```

Only after the fold-0 screen and router pass, score folds 1–4:

```powershell
python src/exp013b_pipeline.py --stage score-qwen-oof --fold 1 --resume
python src/exp013b_pipeline.py --stage score-qwen-oof --fold 2 --resume
python src/exp013b_pipeline.py --stage score-qwen-oof --fold 3 --resume
python src/exp013b_pipeline.py --stage score-qwen-oof --fold 4 --resume
python src/exp013b_pipeline.py --stage evaluate-oof
```

Only after the OOF gate passes:

```powershell
python src/exp013b_pipeline.py --stage train-final
python src/exp013b_pipeline.py --stage build-query-memory --split public
python src/exp013b_pipeline.py --stage build-candidates --split public
python src/exp013b_pipeline.py --stage score-public --resume
python src/exp013b_pipeline.py --stage public-submission
```

`--resume` reuses already complete Qwen query scores in the current stage output.
Use `python src/exp013b_pipeline.py --stage status` to inspect all recorded stage states.
