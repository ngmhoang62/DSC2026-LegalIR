from __future__ import annotations

import gc
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

from .contracts import Progress, digest, read, rrf, seed_all, sha, write
from .learning import ce_loss, checkpoint, local_snapshot, replay_backward, select_negatives, set_rng
from .evidence import Evidence


class CrossEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.snapshot = local_snapshot("BAAI/bge-reranker-v2-m3")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), local_files_only=True)
        base = AutoModelForSequenceClassification.from_pretrained(str(self.snapshot), local_files_only=True, torch_dtype=torch.float32)
        self.model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                                   target_modules=["query", "value"], modules_to_save=["classifier"], bias="none"))
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.to("cuda")
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.load_state_dict(state["adapter"], strict=False)

    def forward(self, pairs):
        batch = self.tokenizer([p["question"] for p in pairs], [p["document"] for p in pairs],
                               padding=True, truncation=False, return_tensors="pt").to("cuda")
        if batch["input_ids"].shape[1] > 512:
            raise ValueError("CE pair budget violated")
        return self.model(**batch).logits.float().reshape(-1)

    def adapter_state(self):
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}


def train_ce(data, store, qids, directory, *, replay=True, max_groups=None):
    from transformers import get_cosine_schedule_with_warmup
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    qids = [q for q in qids if data.gold[q]]
    signature = digest([data.fingerprint, qids, replay, "ce-absolute-v1", sha(Path(__file__)) if not max_groups else 'benchmark'])
    if max_groups and (directory / "BENCHMARK.json").exists():
        value = read(directory / "BENCHMARK.json")
        if value["signature"] != signature:
            raise ValueError("CE benchmark contract changed")
        return value
    success = directory / "_SUCCESS.json"
    if success.exists():
        if read(success)["signature"] != signature:
            raise ValueError("CE resume signature mismatch")
        if read(success)['model_sha256'] != sha(directory/'model.pt'):
            raise ValueError('CE completed model hash mismatch')
        return read(success)
    seed_all()
    model = CrossEncoder(); model.train()
    evidence = Evidence(data, model.tokenizer)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=.01)
    scheduler = get_cosine_schedule_with_warmup(opt, max(1, int(.1*len(qids))), len(qids))
    order = list(qids); random.Random(112).shuffle(order)
    resume = directory / "resume.pt"; position = 0
    if resume.exists():
        receipt = resume.with_suffix('.sha.json')
        if not receipt.exists() or read(receipt)['sha256'] != sha(resume):
            raise ValueError('CE resume checkpoint lacks verified hash')
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("CE checkpoint mismatch")
        model.load_state_dict(state["adapter"], strict=False)
        opt.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        set_rng(state["rng"]); position = state["position"]
    progress = Progress("ce_"+directory.name, len(qids)); begin = time.monotonic()
    try:
        for i in range(position, len(order)):
            q = order[i]; sources = store.rankings(q)
            ranked, _ = rrf([sources[s][:100] for s in ("e5", "lal", "bm25")])
            positives = sorted(data.gold[q])
            negatives = select_negatives(ranked, positives, sources, data.doc_ids, q, 0, 6)
            pairs = [evidence.package(q, d) for d in positives+negatives]
            opt.zero_grad(set_to_none=True)
            if replay:
                loss = replay_backward(model, [[p] for p in pairs], len(positives))
            else:
                value = ce_loss(model(pairs), len(positives)); value.backward(); loss = float(value.detach())
            if not math.isfinite(loss):
                raise FloatingPointError('Non-finite CE loss')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); scheduler.step()
            if (i+1) % 32 == 0 or i+1 == len(order):
                checkpoint(resume, model, opt, scheduler, signature=signature, position=i+1)
            progress.update(i+1, loss=loss)
            if max_groups and i+1 >= max_groups:
                checkpoint(resume, model, opt, scheduler, signature=signature, position=i+1)
                value = dict(signature=signature, benchmark_only=True, groups=i+1, seconds=time.monotonic()-begin,
                             seconds_per_group=(time.monotonic()-begin)/(i+1), peak_vram=torch.cuda.max_memory_reserved())
                write(directory / "BENCHMARK.json", value)
                return value
        checkpoint(directory / "model.pt", model, opt, scheduler, signature=signature, position=len(order))
        result = dict(signature=signature, training_qids=qids, seconds=time.monotonic()-begin, model_sha256=sha(directory/"model.pt"))
        write(success, result); return result
    finally:
        evidence.db.close()
        del evidence, model, opt, scheduler
        gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def score_ce(data, rows, checkpoint_path, output, batch_size=4):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model = CrossEncoder(checkpoint_path); model.eval()
    evidence = Evidence(data, model.tokenizer)
    signature = digest([data.fingerprint, sha(checkpoint_path)])
    progress = Progress("ce_score_"+output.name, len(rows)); started = time.monotonic(); pairs_count = 0
    try:
        for i, (q, upstream) in enumerate(rows.items()):
            path = output / f"{q}.json"
            docs = upstream["order"][:50]
            expected = digest([signature, q, docs])
            if path.exists():
                if read(path)["signature"] != expected:
                    raise ValueError("CE score/candidate resume mismatch")
                continue
            scores = []
            for start in range(0, len(docs), batch_size):
                pairs = [evidence.package(q, d) for d in docs[start:start+batch_size]]
                scores.extend(model(pairs).cpu().tolist()); pairs_count += len(pairs)
            write(path, dict(signature=expected, scores=dict(zip(docs, scores))))
            progress.update(i+1)
        return dict(seconds=time.monotonic()-started, pairs=pairs_count)
    finally:
        evidence.db.close(); del evidence, model
        gc.collect(); torch.cuda.empty_cache()
