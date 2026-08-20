# EXP-013: selective late interaction

EXP-013 keeps structural-v3 as the evidence corpus but stops scoring every
candidate with a heavy cross-encoder.  Its one-time index is compact int8 token
vectors (at most 48 anchors per v3 passage); query-time exact MaxSim is limited
to a small document pool.

## First implementation slice

Run from the LegalIR root with the project Python executable:

```powershell
python src/exp013_pipeline.py --stage preflight
python src/exp013_pipeline.py --stage prepare-models --allow-download --trust-remote-code
python src/exp013_pipeline.py --stage encode-colbert-v3 --allow-download --trust-remote-code --batch-size 12
python src/exp013_pipeline.py --stage build-document-prototypes
```

The download and execution of Jina model code are explicit opt-ins. `preflight`
does neither.  `encode-colbert-v3` intentionally does not support resume: a
partial token-vector store cannot safely be used.  It refuses to overwrite a
different index, so delete/rebuild only after reviewing its manifest.

Inspect only persistent stage progress with:

```powershell
python src/exp013_pipeline.py --stage status
```

## Candidate and ranker path

The next commands use the existing EXP-012 source rankings only as frozen BM25
and BGE candidate channels. They do not rerun EXP-012 retrieval or alter its
artifacts.

```powershell
python src/exp013_pipeline.py --stage encode-colbert-queries --split train --trust-remote-code --batch-size 24
python src/exp013_pipeline.py --stage retrieve-colbert-prototypes --split train
python src/exp013_pipeline.py --stage build-query-memory --split train
python src/exp013_pipeline.py --stage build-candidate-union --split train
python src/exp013_pipeline.py --stage audit-candidate-oracle --split train
python src/exp013_pipeline.py --stage score-exact-maxsim --split train
python src/exp013_pipeline.py --stage build-features --split train
```

Only after candidate oracle metrics are reviewed should LambdaMART be run. It
requires the already-declared `lightgbm` dependency:

```powershell
python -m pip install -r requirements.txt
python src/exp013_pipeline.py --stage train-lambdamart-oof --split train
```

`build-query-memory --split train` is fold-isolated: every query excludes the
entire held-out fold, including itself, before it may read any answer label.
`score-exact-maxsim` puts the 1.25 GiB int8 leaf store on GPU once, then scores
only the fused Top-64 parents. Do not run it until the candidate pool passes
its recall/oracle gate.
