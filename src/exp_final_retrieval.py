"""EXP-final command entrypoint. No expensive work occurs on import."""
import argparse
import json
import traceback
import os
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')

os.environ.setdefault("HF_HOME", "C:/Users/nguye/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from exp_final.contracts import CACHE, RESULTS, Progress, read, write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "seal", "validate", "sparse-check", "public-smoke", "jina-public-check", "jina-check", "preflight", "prepare-frozen", "prepare-jina", "run-fold", "run-all", "evaluate-oof", "fit-final", "predict-public", "verify-submission", "status"])
    parser.add_argument("--outer", choices=[f"fold_{i}" for i in range(5)], default="fold_0")
    parser.add_argument("--budget-hours", type=float, default=48)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Preflight only: smoke checks, never creates RESOURCE_LOCK")
    args = parser.parse_args(argv)
    from exp_final import pipeline
    guard = None
    import threading
    import time
    stopped = threading.Event()
    def heartbeat():
        import psutil
        process = psutil.Process()
        while not stopped.wait(45):
            value = dict(pid=os.getpid(),stage=args.stage,outer=args.outer if args.stage=='run-fold' else None,timestamp=time.time(),
                         rss_bytes=process.memory_info().rss,available_ram_bytes=psutil.virtual_memory().available)
            write(RESULTS/f'heartbeat-{os.getpid()}.json',value)
            print('HEARTBEAT '+json.dumps(value),flush=True)
    try:
        if args.stage not in ('status','validate','verify-submission'):
            from filelock import FileLock
            CACHE.mkdir(parents=True,exist_ok=True)
            guard = FileLock(CACHE/('supervisor.lock' if args.stage=='run-all' else 'worker.lock'), timeout=0)
            guard.acquire()
        if args.stage!='status':
            Progress(args.stage).update(0,force=True)
            threading.Thread(target=heartbeat,daemon=True).start()
        if args.stage == "status":
            result = read(RESULTS/"RUN_STATUS.json") if (RESULTS/"RUN_STATUS.json").exists() else {"state": "NOT_STARTED"}
            import psutil
            result["pid_alive"] = psutil.pid_exists(result.get("pid", -1))
        elif args.stage=='seal':
            from exp_final.seal import seal
            result=seal()
        elif args.stage=='sparse-check':
            from exp_final.validation import sparse_check
            result=sparse_check()
        elif args.stage=='public-smoke':
            from exp_final.validation import public_smoke
            result=public_smoke()
        elif args.stage=='jina-public-check':
            from exp_final.data import Data,SourceStore
            from exp_final.jina import prepare_jina
            data,store=Data(),SourceStore()
            started=time.monotonic()
            result=prepare_jina(data,store,list(data.public)[:2])
            result['end_to_end_seconds']=time.monotonic()-started
            result['passed']=all(store.get(q,'jina') is not None for q in list(data.public)[:2])
            write(RESULTS/'JINA_PUBLIC_SMOKE.json',result)
            store.close()
        elif args.stage == 'jina-check':
            from exp_final.data import Data,SourceStore
            from exp_final.jina import prepare_jina
            data,store=Data(),SourceStore()
            result=prepare_jina(data,store,read(RESULTS/'REPRODUCTION_REPORT.json')['qids'],benchmark=True)
            store.close()
        elif args.stage == "validate":
            from exp_final.validation import validate
            result = validate()
        elif args.stage == "preflight":
            result = pipeline.preflight(args.budget_hours, args.quick)
        elif args.stage == "run-all":
            result = pipeline.run_all(args.budget_hours)
        elif args.stage == "run-fold":
            result = pipeline.run_fold(args.outer)
        elif args.stage == "verify-submission":
            from exp_final.contracts import sha
            result = read(RESULTS/"public/SUBMISSION_MANIFEST.json")
            for name, expected in result["files"].items():
                if sha(RESULTS/"public"/name) != expected:
                    raise ValueError("Submission hash mismatch")
        else:
            result = getattr(pipeline, args.stage.replace("-", "_"))()
        if args.stage == 'validate':
            print(json.dumps({'passed':result['passed'],'report':str(RESULTS/'IMPLEMENTATION_VALIDATION.json')}),flush=True)
        else:
            print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.stage not in ('status','validate'):
            Progress(args.stage).update(1,state='COMPLETE_SUBMISSION' if isinstance(result,dict) and result.get('status')=='COMPLETE_SUBMISSION' else 'STAGE_COMPLETE',force=True)
        return 0
    except BaseException as exc:
        traceback.print_exc()
        Progress(args.stage).update(0, state="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED_RUNTIME", force=True, error=repr(exc))
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        stopped.set()
        if guard is not None:
            guard.release()


if __name__ == "__main__":
    raise SystemExit(main())
