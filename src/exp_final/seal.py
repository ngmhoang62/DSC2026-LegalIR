"""Seal validated implementation and measured resource ledger before any outer run."""
import importlib.metadata
from .contracts import *
from .pipeline import choose_profile


def seal():
    required=['INPUT_AUDIT.json','REPRODUCTION_REPORT.json','IMPLEMENTATION_VALIDATION.json',
              'PUBLIC_PATH_SMOKE.json','SPARSE_PATH_BENCHMARK.json','PREFLIGHT_REPORT.json']
    for name in required:
        value=read(RESULTS/name)
        if value.get('passed') is False:
            raise ValueError(f'Failed validation: {name}')
    pre=read(RESULTS/'PREFLIGHT_REPORT.json'); resource=read(RESULTS/'RESOURCE_LOCK.json')
    feature=read(RESULTS/'REAL_FEATURE_BENCHMARK.json'); sparse=read(RESULTS/'SPARSE_PATH_BENCHMARK.json')
    costs=dict(pre['costs'])
    # All three blocks, both families, two outer ML models and one deployment model.
    lm=feature['lambdamart_fit_and_32_query_prediction_seconds']/96
    lr=feature['lr_fit_and_32_query_prediction_seconds']/96
    measured_ml=(lm+lr)*4200*5*3 + max(lm,lr)*(5600*10+7000)
    costs['ml']=max(costs['ml'],measured_ml)+4*3600  # explicit calibrated-ranking/feature I/O reserve
    costs['frozen']+=sparse['seconds_per_query']*2400+300
    jina_ok=False
    if resource['jina']:
        jr=read(CACHE/'JINA_REUSE_VALIDATION.json');jp=read(RESULTS/'JINA_PUBLIC_SMOKE.json')
        if not jr['passed'] or not jp['passed']:
            raise ValueError('Jina reuse/public validation failed')
        jina_ok=True
        # Query repair benchmark is deliberately conservative (includes parity probes).
        costs['jina']=max(costs['jina'],jr['seconds']/jr['queries']*7000 + jp['seconds']/max(1,jp['queries'])*1000)+1200
    chosen,forecasts=choose_profile(costs,48,jina_ok)
    if chosen['profile']!=resource['profile']:
        raise ValueError('Measured supplement changes resource profile; review before replacing preliminary lock')
    write(RESULTS/'JOB_LEDGER_REVISED.json',dict(costs=costs,forecasts=forecasts,selected=chosen,
          safety_multiplier=1.25,recovery_reserve_seconds=7200,scoring_io_reserve_seconds=4*3600,
          caveat='Forecast, not a guaranteed completion time; rolling throughput must be checked.'))
    files=[*list((ROOT/'src/exp_final').glob('*.py')),ROOT/'src/exp_final_retrieval.py',
           ROOT/'tests/test_exp_final_retrieval.py', ROOT/'docs/EXP_FINAL_PLAN.md',
           ROOT/'public_test_dataset/train.json',ROOT/'public_test_dataset/public-official.json',ROOT/'cache/cv_folds.json',
           ROOT/'cache/structural_v3_e5_final_v1/manifest.json',ROOT/'cache/e5_final_v1/manifest.json',
           ROOT/'cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/manifest.json',
           ROOT/'src/exp111_multiview_sparse_retrieval.py',ROOT/'src/exp109b_encoder_complementarity.py',
           ROOT/'src/exp109c_latent_condition_late_interaction.py',ROOT/'src/exp108_atomic_condition_reranker.py']
    models=read(RESULTS/'INPUT_AUDIT.json')['models']
    if resource['jina']:
        from exp109c_latent_condition_late_interaction import local_jina_snapshot
        models['jinaai/jina-colbert-v2']=str(local_jina_snapshot())
    model_files={}
    for folder in models.values():
        for p in Path(folder).iterdir():
            if p.suffix=='.safetensors' or p.name in ('config.json','tokenizer_config.json','tokenizer.json'):
                model_files[str(p)]=sha(p)
    result=dict(status='SEALED_FOR_PRODUCTION',resource_lock_sha256=sha(RESULTS/'RESOURCE_LOCK.json'),
                files={str(p.relative_to(ROOT)):sha(p) for p in files},models=models,model_files=model_files,
                validation_artifacts={n:sha(RESULTS/n) for n in required},
                estimated_seconds=chosen['seconds'],profile=chosen['profile'],
                versions={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','numpy','lightgbm','scikit-learn')},
                parameter_limit_audit=read(RESULTS/'INPUT_AUDIT.json')['conservative_active_parameters'],
                exp110p='CANCELLED_EXCLUDED',outer_folds=5,uploaded=False)
    lock(RESULTS/'RUN_LOCK.json',result)
    return result
