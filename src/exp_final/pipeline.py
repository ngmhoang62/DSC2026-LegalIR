from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .contracts import *
from .data import Data, SourceStore, import_sources, materialize_lal, prepare_sparse, public_dense
from .learning import QueryEncoder, local_snapshot, train_query, score_queries
from .fusion import feature_names, fit_ranker, upstream
from .cross_encoder import train_ce, score_ce


def audit():
    if not (RESULTS / "READING_AUDIT.json").exists():
        raise ValueError("Reading audit required before execution")
    data = Data()
    if len(data.doc_ids) != 8507 or len(data.chunk_ids) != 343347 or len(data.train) != 7000 or data.label_audit["evaluable_queries"] != 6991:
        raise ValueError("Canonical counts changed")
    if len(set(q for rows in data.folds.values() for q in rows)) != 7000:
        raise ValueError("Fold coverage invalid")
    if set(data.public) & set(data.train):
        raise ValueError("Train/public qid collision")
    for outer in data.folds:
        splits(data.folds, outer)
    manifest = read(ROOT / "cache/e5_final_v1/manifest.json")
    for name, expected in manifest["artifact_sha256"].items():
        if sha(ROOT / "cache/e5_final_v1" / name) != expected:
            raise ValueError("Frozen E5 artifact hash mismatch")
    models = {repo: str(local_snapshot(repo)) for repo in ("mainguyen9/vietlegal-e5", "darklethelong/vnlegal-lal", "BAAI/bge-reranker-v2-m3")}
    from safetensors import safe_open
    parameter_counts = {}
    for repo, folder in models.items():
        count = 0
        for weight in Path(folder).glob("*.safetensors"):
            with safe_open(str(weight), framework="pt", device="cpu") as f:
                count += sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())
        if not count:
            raise ValueError(f"Cannot audit parameters for {repo}")
        parameter_counts[repo] = count
    # Conservative: count both E5 roles separately, Jina and generous adapter/head reserve.
    conservative_parameters = sum(parameter_counts.values()) + parameter_counts["mainguyen9/vietlegal-e5"] + 559497216 + 50000000
    if conservative_parameters > 4_000_000_000:
        raise ValueError("Active parameter budget exceeded")
    schema = {f"B{b}": feature_names(b) for b in range(3)}
    write(RESULTS / "FEATURE_SCHEMA.json", schema)
    result = dict(status="PASS_INPUT_AUDIT", data_fingerprint=data.fingerprint, labels=data.label_audit,
                  documents=len(data.doc_ids), chunks=len(data.chunk_ids), public_queries=len(data.public),
                  models=models, parameter_counts=parameter_counts, conservative_active_parameters=conservative_parameters,
                  exp110p="USER_CANCELLED_EXCLUDED", performance_stop_gates=False,
                  official_id_restricted_oracle=float(np.mean([len(g & set(data.doc_ids))/len(g) for g in data.original.values()])))
    write(RESULTS / "INPUT_AUDIT.json", result)
    return result


def real_reproduction(data, selected):
    from .learning import ParentBank
    model = QueryEncoder(); model.eval()
    with torch.no_grad():
        observed = model([data.questions[q] for q in selected[:8]]).cpu().numpy()
    expected = np.stack([data.query_vector(q, "e5") for q in selected[:8]])
    error = float(np.max(np.abs(observed-expected)))
    chunk_indices = np.concatenate(data.positions[:20])
    parent = np.concatenate([np.full(len(p), i) for i, p in enumerate(data.positions[:20])])
    v = np.array(data.matrix("e5")[chunk_indices], dtype=np.float32)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    bank = ParentBank(v, parent, device="cuda")
    scores, _ = bank.mine(torch.tensor(observed, device="cuda"))
    reference = np.stack([np.stack([np.sort(v[parent==p]@q)[-2:].mean() for p in range(20)]) for q in observed])
    parent_error = float(np.max(np.abs(scores.cpu().numpy()-reference)))
    report = dict(identity_query_max_abs_error=error, real_parent_max_abs_error=parent_error,
                  tolerance=1e-5, qids=selected[:8], fixture_hash=digest([chunk_indices.tolist(), selected[:8]]),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    report["passed"] = error <= 1e-5 and parent_error <= 1e-5
    write(RESULTS/"REPRODUCTION_REPORT.json", report)
    del model, bank; gc.collect(); torch.cuda.empty_cache()
    if not report["passed"]:
        raise ValueError("Real reproduction mismatch; inspect per-query numerical path")
    return report


def prepare_frozen(qids=None):
    data = Data(); store = SourceStore()
    materialize_lal(data)
    import_sources(store)
    prepare_sparse(data, store, qids or list(data.questions))
    if qids is None:
        for source in ("e5", "lal"):
            public_dense(data, store, source)
    store.close()


def choose_profile(costs, budget_hours=48, jina_eligible=True):
    profiles = [("P0", 2, True, 2), ("P1", 2, True, 1), ("P2", 2, False, 1),
                ("P3", 2, True, 0), ("P4", 2, False, 0), ("P5", 1, False, 0)]
    forecasts = []
    for name, epochs, jina, ce in profiles:
        if jina and not jina_eligible:
            continue
        raw = costs["frozen"] + costs["query_epoch"]*epochs + costs["query_scoring"] + costs["ml"] + costs["public"]
        raw += costs["jina"] if jina else 0.
        raw += costs["ce_inner_final"] + costs["ce_scoring"] if ce else 0.
        raw += costs["ce_refit"] if ce == 2 else 0.
        forecast = raw*1.25 + 2*3600
        forecasts.append(dict(profile=name, seconds=forecast, epochs=epochs, jina=jina, ce=ce))
    for ceiling in (budget_hours, 50):
        for p in forecasts:
            if p["seconds"] <= ceiling*3600:
                return p, forecasts
    raise RuntimeError("BUDGET_EXTENSION_REQUIRED " + json.dumps(forecasts))


def preflight(budget_hours=48, quick=False):
    import psutil
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    info = audit()
    data = Data(); store = SourceStore()
    t, c, f = splits(data.folds, "fold_0")
    qids = sorted(t, key=lambda q: len(data.questions[q]))
    selected = [qids[i] for i in np.linspace(0, len(qids)-1, 256 if not quick else 16, dtype=int)]
    started = time.monotonic()
    materialize_lal(data); import_sources(store); prepare_sparse(data, store, selected)
    source_seconds = time.monotonic()-started
    reproduction = real_reproduction(data, selected)
    attempts = []
    query = None
    for micro in (4, 2, 1):
        torch.cuda.reset_peak_memory_stats()
        try:
            query = train_query(data, store, selected, CACHE / "preflight" / f"query-m{micro}-{'quick' if quick else 'full'}",
                                epochs=8 if not quick else 1, nominal_epochs=8 if not quick else 1,
                                microbatch=micro, max_updates=128 if not quick else 1)
            attempts.append(dict(microbatch=micro, **query))
            if query.get("peak_vram", 0) > .9*torch.cuda.get_device_properties(0).total_memory:
                raise torch.cuda.OutOfMemoryError("Less than ten percent allocator headroom")
            break
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append(dict(microbatch=micro, oom=str(exc))); gc.collect(); torch.cuda.empty_cache()
            query = None
    if query is None:
        raise RuntimeError("FP32 query LoRA preflight failed; FP16 requires a separately tested numerical path")
    # CE direct-group trial first, then exact logit-gradient replay on measured OOM.
    ce = None
    for replay in (False, True):
        torch.cuda.reset_peak_memory_stats()
        try:
            groups = sorted(selected, key=lambda q: (-len(data.gold[q]), q))[:64 if not quick else 1]
            ce = train_ce(data, store, groups, CACHE / "preflight" / f"ce-{replay}-{'quick' if quick else 'full'}", replay=replay, max_groups=len(groups))
            if ce.get("peak_vram", 0) > .9*torch.cuda.get_device_properties(0).total_memory:
                raise torch.cuda.OutOfMemoryError("CE headroom below ten percent")
            break
        except torch.cuda.OutOfMemoryError:
            ce = None; gc.collect(); torch.cuda.empty_cache()
    result = dict(status="PASS_QUICK_ONLY" if quick else "BENCHMARK_IN_PROGRESS", query=query, query_attempts=attempts,
                  ce=ce, ce_replay=replay, source_seconds=source_seconds, available_ram=psutil.virtual_memory().available,
                  quick=quick, microbatch=micro, reproduction=reproduction)
    if quick:
        write(RESULTS / "QUICK_PREFLIGHT.json", result); store.close(); return result
    query_ckpt = CACHE / "preflight" / f"query-m{micro}-full" / "resume.pt"
    score_begin = time.monotonic()
    score_queries(data, selected, query_ckpt, CACHE / "preflight/query-scores")
    score_seconds = time.monotonic()-score_begin
    rows = {q: read(CACHE / "preflight/query-scores" / f"{q}.json") for q in selected[:64]}
    ce_score = dict(seconds=0., pairs=0)
    if ce:
        ce_score = score_ce(data, rows, CACHE / "preflight" / f"ce-{replay}-full/resume.pt", CACHE / "preflight/ce-scores")
    jina_eligible, jina_cost = False, 0.
    try:
        from .jina import prepare_jina
        jb = prepare_jina(data, store, selected[:8], benchmark=True)
        jina_cost = jb["seconds"] / max(1, jb["queries"]) * (len(data.questions)-8)
        jina_eligible = True
        result["jina_benchmark"] = jb
    except (OSError, ValueError, RuntimeError) as exc:
        result["jina_unavailable"] = repr(exc)
    qepoch_groups = sum(math.ceil(len([q for q in splits(data.folds, o)[0] if data.gold[q]])/16) + math.ceil(len([q for q in data.train if q not in data.folds[o] and data.gold[q]])/16) for o in data.folds) + math.ceil(6991/16)
    ce_group = ce["seconds_per_group"] if ce else 1e12
    costs = dict(frozen=source_seconds/len(selected)*(len(data.questions)-len(selected)), query_epoch=qepoch_groups*query["seconds_per_update"],
                 query_scoring=score_seconds/len(selected)*(5*1400*2+7000+1000), ml=2*3600., public=2*3600., jina=jina_cost,
                 ce_inner_final=(5*4200+6991)*ce_group, ce_refit=5*5600*ce_group,
                 ce_scoring=ce_score["seconds"]/max(1, ce_score["pairs"])*(5*2800+1000)*50 if ce else 1e12)
    profile, forecasts = choose_profile(costs, budget_hours, jina_eligible)
    result.update(status="PASS_PREFLIGHT", costs=costs, forecasts=forecasts, selected=profile)
    write(RESULTS / "PREFLIGHT_REPORT.json", result)
    write(RESULTS / "JOB_LEDGER.json", dict(costs=costs, forecasts=forecasts, completion_reserve_seconds=7200))
    lock(RESULTS / "RESOURCE_LOCK.json", {**profile, "microbatch": micro, "ce_replay": replay, "data_fingerprint": data.fingerprint})
    store.close(); return result


def subset(mapping, ids):
    return {q: mapping[q] for q in ids}


def metric_rows(rows, data):
    p = {q: r["order"] for q, r in rows.items()}
    return report_metrics(p, subset(data.original, p), subset(data.gold, p))


def run_fold(outer):
    data = Data(); store = SourceStore(); resource = read(RESULTS / "RESOURCE_LOCK.json")
    directory = RESULTS / "outer" / outer; directory.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "outer" / outer
    train, cal, test = splits(data.folds, outer)
    completed = directory / "REPORT.json"
    if completed.exists():
        existing = read(directory/'PREDICTION_LOCK.json')
        if existing['predictions_hash'] != digest(read(directory/'PREDICTIONS.json')) or existing['selection_sha256'] != sha(directory/'SELECTION_LOCK.json'):
            raise ValueError('Completed fold lock/hash mismatch')
        return read(completed)
    epochs = resource["epochs"]
    selection_path = directory / "SELECTION_LOCK.json"
    models = {}
    if not selection_path.exists():
        train_query(data, store, train, cache/"query-inner", epochs=epochs, nominal_epochs=epochs, microbatch=resource["microbatch"])
        for epoch in range(1, epochs+1):
            score_queries(data, cal, cache/"query-inner"/f"epoch-{epoch}.pt", cache/f"cal-query-{epoch}")
        adapted = {e: DiskRows(cache/f'cal-query-{e}',cal) for e in range(1,epochs+1)}
        recipes = [dict(family="rrf", block=0, epoch=0, beta=0., pool="frozen", alpha=0., confidence=False)]
        for block in range(3 if resource["jina"] else 2):
            for family in ("lm", "lr"):
                models[family, block] = fit_ranker(data, store, train, family, block, cache/f"inner-{family}-b{block}.pkl")
                recipes.append(dict(family=family, block=block, epoch=0, beta=0., pool="frozen", alpha=0., confidence=False))
                for epoch in range(1, epochs+1):
                    for beta in (0., .15, .3, .5, 1.):
                        recipes.append(dict(family=family, block=block, epoch=epoch, beta=beta, pool="expanded", alpha=0., confidence=False))
        best, best_rows, best_key = None, None, None
        anchor_recipe, anchor_key = None, None
        screens = []
        for recipe in recipes:
            model = models.get((recipe["family"], recipe["block"]))
            rows = {q: upstream(data, store, q, model, recipe, adapted.get(recipe["epoch"], {}).get(q)) for q in cal}
            m = metric_rows(rows, data); key = metric_key(m)
            screens.append(dict(recipe=recipe, metrics=m))
            if recipe["pool"] == "frozen" and (anchor_key is None or key > anchor_key):
                anchor_recipe, anchor_key = dict(recipe), key
            if best_key is None or key > best_key:
                best, best_rows, best_key = dict(recipe), rows, key
        write(directory/"CALIBRATION_SCREEN.json", screens)
        if resource["ce"]:
            train_ce(data, store, train, cache/"ce-inner", replay=resource["ce_replay"])
            score_ce(data, best_rows, cache/"ce-inner/model.pt", cache/"cal-ce")
            confidence = [float(zscore(r["scores"][:50])[4]-zscore(r["scores"][:50])[5]) for r in best_rows.values() if len(r["order"])>5]
            threshold = float(np.quantile(confidence, .7)) if confidence else 0.
            base_rows = best_rows
            for alpha in (.1, .25):
                for conf in (False, True):
                    candidate = {}
                    for q, r in base_rows.items():
                        order, scores = correct(r["order"], r["scores"], read(cache/"cal-ce"/f"{q}.json")["scores"], alpha, threshold if conf else None)
                        candidate[q] = dict(order=order, scores=scores)
                    key = metric_key(metric_rows(candidate, data))
                    if key > best_key:
                        best_key, best_rows = key, candidate
                        best.update(alpha=alpha, confidence=conf, threshold=threshold if conf else None)
        best["adaptive_threshold"] = select_threshold(best_rows, subset(data.original, cal))
        best["nominal_epochs"] = epochs
        best["anchor_recipe"] = anchor_recipe
        best["scope"] = dict(inner_train=train, calibration=cal, outer=test)
        best["data_fingerprint"] = data.fingerprint
        lock(selection_path, best)
    selected = read(selection_path)
    if set(selected["scope"]["inner_train"]) & set(test) or set(selected["scope"]["calibration"]) & set(test):
        raise ValueError("Selection leakage")
    outer_train = train+cal
    diagnostic_epoch = selected['epoch'] or epochs
    train_query(data, store, outer_train, cache/"query-outer", epochs=diagnostic_epoch, nominal_epochs=epochs, microbatch=resource["microbatch"])
    score_queries(data, test, cache/"query-outer"/f"epoch-{diagnostic_epoch}.pt", cache/"test-query")
    model = None
    if selected["family"] != "rrf":
        model = fit_ranker(data, store, outer_train, selected["family"], selected["block"], cache/"outer-ml.pkl")
    rows = {q: upstream(data, store, q, model, selected, read(cache/"test-query"/f"{q}.json") if selected["epoch"] else None) for q in test}
    anchor_recipe = selected["anchor_recipe"]
    anchor_model = None
    if anchor_recipe["family"] != "rrf":
        anchor_model = fit_ranker(data, store, outer_train, anchor_recipe["family"], anchor_recipe["block"], cache/"outer-anchor.pkl")
    anchors = {q: upstream(data, store, q, anchor_model, anchor_recipe) for q in test}
    write(directory/"ANCHOR_PREDICTIONS.json", anchors)
    write(directory/"UPSTREAM_PREDICTIONS.json", rows)
    if selected["alpha"]:
        ce_folder = cache/"ce-inner"
        if resource["ce"] == 2:
            ce_folder = cache/"ce-outer"
            train_ce(data, store, outer_train, ce_folder, replay=resource["ce_replay"])
        score_ce(data, rows, ce_folder/"model.pt", cache/"test-ce")
        for q, r in rows.items():
            order, scores = correct(r["order"], r["scores"], read(cache/"test-ce"/f"{q}.json")["scores"], selected["alpha"], selected.get("threshold") if selected["confidence"] else None)
            rows[q] = dict(order=order, scores=scores)
    lock(directory/"PREDICTION_LOCK.json", dict(selection_sha256=sha(selection_path), predictions_hash=digest(rows), qids=test))
    write(directory/"PREDICTIONS.json", rows)
    adaptive = {q: prefix(r["order"], r["scores"], selected["adaptive_threshold"]) for q, r in rows.items()}
    lost = sum(len((set(r["order"][:5]) & data.original[q])-set(adaptive[q])) for q, r in rows.items())
    result = dict(outer=outer, status="COMPLETE_FOLD", metrics=metric_rows(rows, data), anchor=metric_rows(anchors, data),
                  upstream_metrics=metric_rows(read(directory/"UPSTREAM_PREDICTIONS.json"), data),
                  adaptive=report_metrics(adaptive, subset(data.original, test), subset(data.gold, test)), adaptive_lost_hits=lost,
                  selection=selected, performance_stop=False)
    from .diagnostics import ranking_diagnostics
    result['diagnostics'] = ranking_diagnostics(data, store, rows, anchors, DiskRows(cache/'test-query',test))
    result['diagnostic_adapter_epoch'] = diagnostic_epoch
    write(completed, result); store.close(); return result


def evaluate_oof():
    data = Data(); reports, rows, selections, anchors = [], {}, [], {}
    for outer in data.folds:
        directory = RESULTS/"outer"/outer
        result, local = read(directory/"REPORT.json"), read(directory/"PREDICTIONS.json")
        if set(rows) & set(local) or set(local) != set(data.folds[outer]):
            raise ValueError("OOF overlap/incomplete fold")
        prediction_lock = read(directory/"PREDICTION_LOCK.json")
        if digest(local) != prediction_lock["predictions_hash"]:
            raise ValueError("Locked predictions changed")
        rows.update(local); reports.append(result); selections.append(result["selection"])
        anchors.update(read(directory/"ANCHOR_PREDICTIONS.json"))
    if set(rows) != set(data.train):
        raise ValueError("OOF qid coverage mismatch")
    measured = metric_rows(rows, data)
    values = [r["metrics"]["official"]["recall@5"] for r in reports]
    deltas = np.array([(len(set(rows[q]["order"][:5]) & data.original[q])-len(set(anchors[q]["order"][:5]) & data.original[q]))/len(data.original[q]) for q in rows])
    rng = np.random.default_rng(112)
    boot = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(2000)])
    result = dict(status="COMPLETE_CV", folds=reports, pooled=measured, mean_recall=float(np.mean(values)), std_recall=float(np.std(values)),
                  public_score=None, historical_selection_bias=True,
                  paired_delta_bootstrap=dict(mean=float(deltas.mean()), lower=float(np.quantile(boot, .025)), upper=float(np.quantile(boot, .975))),
                  query_wins=int((deltas>0).sum()), query_losses=int((deltas<0).sum()))
    write(RESULTS/"OOF_REPORT.json", result)
    recipe = deployment_recipe(selections)
    finite = all(x["adaptive_threshold"] is not None for x in selections)
    adaptive = finite and measured["official"]["recall@5"] >= .98 and all(r["adaptive_lost_hits"] == 0 for r in reports)
    recipe["deployment_adaptive_threshold"] = max(x["adaptive_threshold"] for x in selections) if adaptive else None
    recipe["scope"] = {"training": list(data.train), "evaluation": "public_without_labels"}
    lock(RESULTS/"FINAL_CONFIG_LOCK.json", recipe)
    return result


def fit_final():
    data = Data(); store = SourceStore(); recipe = read(RESULTS/"FINAL_CONFIG_LOCK.json"); resource = read(RESULTS/"RESOURCE_LOCK.json")
    directory = CACHE/"final_full_train"; qids = list(data.train)
    if recipe["epoch"]:
        train_query(data, store, qids, directory/"query", epochs=recipe["epoch"], nominal_epochs=recipe["nominal_epochs"], microbatch=resource["microbatch"])
    if recipe["family"] != "rrf":
        fit_ranker(data, store, qids, recipe["family"], recipe["block"], directory/"ml.pkl")
    if recipe["alpha"]:
        train_ce(data, store, qids, directory/"ce", replay=resource["ce_replay"])
    store.close()


def predict_public():
    import pickle
    data = Data(); store = SourceStore(); recipe = read(RESULTS/"FINAL_CONFIG_LOCK.json")
    directory = CACHE/"final_full_train"; public = CACHE/"public"
    if recipe["epoch"]:
        score_queries(data, list(data.public), directory/"query"/f"epoch-{recipe['epoch']}.pt", public/"query")
    model = None
    if recipe["family"] != "rrf":
        with (directory/"ml.pkl").open("rb") as f:
            model = pickle.load(f)
    rows = {q: upstream(data, store, q, model, recipe, read(public/"query"/f"{q}.json") if recipe["epoch"] else None) for q in data.public}
    if recipe["alpha"]:
        score_ce(data, rows, directory/"ce/model.pt", public/"ce")
        for q, r in rows.items():
            order, scores = correct(r["order"], r["scores"], read(public/"ce"/f"{q}.json")["scores"], recipe["alpha"], recipe.get("threshold") if recipe["confidence"] else None)
            rows[q] = dict(order=order, scores=scores)
    write(RESULTS/"public/PREDICTIONS.json", rows)
    fixed = {q: r["order"][:5] for q, r in rows.items()}
    chosen = fixed
    if recipe["deployment_adaptive_threshold"] is not None:
        adaptive = {q: {"answer": prefix(r["order"], r["scores"], recipe["deployment_adaptive_threshold"])} for q, r in rows.items()}
        write(RESULTS/"public/submission_adaptive_diagnostic.json", adaptive)
        chosen = {q: r["answer"] for q, r in adaptive.items()}
    result = export_submission(RESULTS/"public", chosen, data.public, data.doc_ids, recall_first=fixed)
    store.close(); return result


def run_all(budget_hours=48):
    import subprocess
    import sys
    run_lock=read(RESULTS/'RUN_LOCK.json')
    for name,expected in run_lock['files'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError(f'Frozen run input/code changed: {name}')
    for name,expected in run_lock.get('model_files',{}).items():
        if sha(Path(name))!=expected:
            raise ValueError(f'Pinned model file changed: {name}')
    def child(stage, extra=()):
        for name,expected in run_lock['files'].items():
            if sha(ROOT/name)!=expected:
                raise ValueError(f'Frozen run input/code changed before {stage}: {name}')
        logs = RESULTS/"logs"; logs.mkdir(parents=True, exist_ok=True)
        name = stage + ("-"+"-".join(extra) if extra else "")
        with (logs/f"{name}.stdout.log").open("a", encoding="utf-8") as out, (logs/f"{name}.stderr.log").open("a", encoding="utf-8") as err:
            command = [sys.executable, "-u", str(ROOT/"src/exp_final_retrieval.py"), stage, "--resume", *extra]
            process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
            print(f"stage={stage} child_pid={process.pid} log={out.name}", flush=True)
            started, last_heartbeat = time.monotonic(), 0.
            while process.poll() is None:
                if time.monotonic()-last_heartbeat >= 45:
                    import psutil
                    last_heartbeat = time.monotonic()
                    value = dict(supervisor_pid=os.getpid(), child_pid=process.pid, stage=stage, arguments=list(extra),
                                 state='RUNNING', elapsed_seconds=last_heartbeat-started, timestamp=time.time(),
                                 available_ram_bytes=psutil.virtual_memory().available, stdout=str(out.name), stderr=str(err.name))
                    write(RESULTS/'SUPERVISOR_STATUS.json', value)
                    print(json.dumps(value), flush=True)
                time.sleep(5)
            value.update(state='FAILED' if process.returncode else 'STAGE_COMPLETE',exit_code=process.returncode,timestamp=time.time())
            write(RESULTS/'SUPERVISOR_STATUS.json', value)
            if process.returncode:
                raise RuntimeError(f"Child {stage} failed exit={process.returncode}; inspect {err.name}")
    child("audit")
    if not (RESULTS/'IMPLEMENTATION_VALIDATION.json').exists() or not read(RESULTS/'IMPLEMENTATION_VALIDATION.json').get('passed'):
        raise ValueError('Real integration validation required before production dispatch')
    if not (RESULTS/"RESOURCE_LOCK.json").exists():
        child("preflight", ("--budget-hours", str(budget_hours)))
    child("prepare-frozen")
    if read(RESULTS/"RESOURCE_LOCK.json")["jina"]:
        child("prepare-jina")
    for outer in (f"fold_{i}" for i in range(5)):
        child("run-fold", ("--outer", outer))
    for stage in ("evaluate-oof", "fit-final", "predict-public", "verify-submission"):
        child(stage)
    result = read(RESULTS/"public/SUBMISSION_MANIFEST.json")
    write(RESULTS/'SUPERVISOR_STATUS.json', dict(state='COMPLETE_SUBMISSION',supervisor_pid=os.getpid(),timestamp=time.time()))
    Progress("complete", 1).update(1, state="COMPLETE_SUBMISSION", force=True)
    return result


def prepare_jina():
    from .jina import prepare_jina as build
    data = Data(); store = SourceStore()
    result = build(data, store, list(data.questions))
    store.close(); return result
